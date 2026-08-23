#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include <cpuid.h>
#include <omp.h>
#include <immintrin.h>

#define MOG_MAGIC "MOG\0"
#define MOG_VERSION 10
#define TOKEN_MAX 94
#define VOCAB_MAX 262144
#define HIDDEN_SIZE 1536
#define SPECIAL_MAX 256
#define ENCODE_VOCAB_MAX 32768
#define MERGES_MAX 514906
#define NUM_LAYERS 35
#define NUM_KV_CACHE_OWNERS 15
#define TENSOR_MAX (2 + NUM_LAYERS * 19 + 4)
#define BATCH_SEQ 512
#define PACKED_ROWS 16

// ----------------------------------------------------------------------------
// Config

typedef struct {
    int max_seq_len;
    float norm_eps;
    char layer_types[64];
    int sliding_window;
    int hidden_size_per_layer_input;
    float final_logit_softcapping;
} Config;

// ----------------------------------------------------------------------------
// Tokenizer types

typedef struct {
    int left, right, result, rank;
} Merge;

typedef struct {
    char token[TOKEN_MAX];
    int id;
} VocabEntry;

typedef struct {
    int n_merges;
    int n_encode_vocab;
    int n_specials;
    char decoded_tokens[VOCAB_MAX][TOKEN_MAX];
    VocabEntry specials[SPECIAL_MAX];
    VocabEntry encode_vocab[ENCODE_VOCAB_MAX];
    Merge merges[MERGES_MAX];
} Tokenizer;

// ----------------------------------------------------------------------------
// Model

typedef struct {
    void *data;
    uint16_t *scales;
    int shape[4];
} Tensor;

typedef struct {
    Tensor input_layernorm;
    Tensor layer_scalar;
    Tensor pre_ffn_layernorm;
    Tensor post_attn_layernorm;
    Tensor post_ffn_layernorm;
    Tensor post_per_layer_input_norm;
    Tensor per_layer_input_gate;
    Tensor per_layer_projection;
    Tensor q_norm;
    Tensor k_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor rope_cos;
    Tensor rope_sin;
} Gemma4LayerWeights;

typedef struct {
    Tensor embed;
    Tensor embed_per_layer;
    Gemma4LayerWeights layers[NUM_LAYERS];
    Tensor norm;
    Tensor per_layer_model_projection;
    Tensor per_layer_projection_norm;
    Tensor gelu_table;
} Gemma4Weights;

typedef struct {
    float  *residual;                       // Hidden states carried through residual connections.
    float  *hidden;                         // Primary temporary input and output for each sublayer.
    float  *branch;                         // Query or MLP up projection kept alongside hidden.
    int8_t *xq;                             // Quantized input rows consumed by matmul_int8.
    float  *xq_scales;                      // Quantization scales for xq.
    float  *logits;                         // Vocabulary scores.
    float  *kv_cache[NUM_LAYERS];           // Cached keys and values; shared layers alias their owner.
    int     kv_dim[NUM_LAYERS];             // Key/value row width for each layer.
    float  *per_layer;                      // Token-conditioned input for each layer.
} RunState;

typedef struct {
    char magic[4];
    uint32_t version;
    Config config;
    Tokenizer tokenizer;
    Gemma4Weights weights;
} Gemma4Model;

// ----------------------------------------------------------------------------
// Loader

_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8 && sizeof(Config) == 84 && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96 && sizeof(Tokenizer) == 36182444 && sizeof(Tensor) == 32 && sizeof(Gemma4Weights) == 21472 && sizeof(Gemma4Model) == 36204008 && offsetof(Gemma4Model, weights) == 36182536, "MOG ABI mismatch");

int load_model(Gemma4Model *m) {
    if (memcmp(m->magic, MOG_MAGIC, 4) != 0 || m->version != MOG_VERSION)
        return -1;

    Tensor *t = (Tensor *)&m->weights;
    for (int i = 0; i < TENSOR_MAX; i++) {
        t[i].data = t[i].data ? (void *)((uint8_t *)m + (uintptr_t)t[i].data) : NULL;
        t[i].scales = t[i].scales ? (uint16_t *)((uint8_t *)m + (uintptr_t)t[i].scales) : NULL;
    }

    return 0;
}

void *alloc(uint8_t **cursor, size_t size) {
    void *pointer = *cursor;
    *cursor += size;
    return pointer;
}

void init_run_state(RunState *s, const Config *c, const Gemma4Weights *w) {
    int intermediate_max = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        if (w->layers[l].gate_proj.shape[0] > intermediate_max) intermediate_max = w->layers[l].gate_proj.shape[0];
    }

    uint8_t *cursor = calloc(1, 6ULL << 30);
    s->residual = alloc(&cursor, (size_t)BATCH_SEQ * HIDDEN_SIZE * sizeof(float));
    s->hidden = alloc(&cursor, (size_t)BATCH_SEQ * intermediate_max * sizeof(float));
    s->branch = alloc(&cursor, (size_t)BATCH_SEQ * intermediate_max * sizeof(float));
    s->xq = alloc(&cursor, (size_t)BATCH_SEQ * intermediate_max * sizeof(int8_t));
    s->xq_scales = alloc(&cursor, (size_t)BATCH_SEQ * intermediate_max / 64 * sizeof(float));
    s->logits = alloc(&cursor, (size_t)VOCAB_MAX * sizeof(float));
    s->per_layer = alloc(&cursor, (size_t)BATCH_SEQ * NUM_LAYERS * c->hidden_size_per_layer_input * sizeof(float));
    for (int l = 0; l < NUM_KV_CACHE_OWNERS; l++) {
        int kv_dim = w->layers[l].k_proj.shape[0];
        s->kv_cache[l] = alloc(&cursor, 2 * (size_t)c->max_seq_len * kv_dim * sizeof(float));
        s->kv_dim[l] = kv_dim;
    }
    for (int l = NUM_KV_CACHE_OWNERS; l < NUM_LAYERS; l++) {
        int owner = NUM_KV_CACHE_OWNERS - 1;
        while (c->layer_types[owner] != c->layer_types[l]) owner--;
        s->kv_cache[l] = s->kv_cache[owner];
        s->kv_dim[l] = s->kv_dim[owner];
    }
}

// ----------------------------------------------------------------------------
// Tokenizer

int cmp_vocab(const void *key, const void *elem) {
    return strcmp((const char *)key, ((const VocabEntry *)elem)->token);
}

int cmp_merge(const void *key, const void *elem) {
    const int *k = key;
    const Merge *m = elem;
    if (k[0] != m->left) return k[0] - m->left;
    return k[1] - m->right;
}

const VocabEntry *find_token(const Tokenizer *t, const char *token) {
    return bsearch(token, t->encode_vocab, t->n_encode_vocab, sizeof(VocabEntry), cmp_vocab);
}

int bpe(const Tokenizer *t, int *ids, int n) {
    for (;;) {
        const Merge *best = NULL;
        int where = -1;
        for (int i = 0; i + 1 < n; i++) {
            const Merge *m = bsearch(ids + i, t->merges, t->n_merges, sizeof(Merge), cmp_merge);
            if (m && (!best || m->rank < best->rank)) { best = m; where = i; }
        }
        if (!best) return n;

        ids[where] = best->result;
        memmove(ids + where + 1, ids + where + 2, (n - where - 2) * sizeof(*ids));
        n--;
    }
}

int encode(const Tokenizer *t, const char *text, int *ids, int cap) {
    int n = 1;

    for (const char *p = text; *p;) {
        if (n >= cap) return -1;
        int special = -1;
        if (*p == '<')
            for (int i = 0; i < t->n_specials && special < 0; i++) {
                int len = (int)strlen(t->specials[i].token);
                if (!strncmp(p, t->specials[i].token, len)) { special = t->specials[i].id; p += len; }
            }
        if (special >= 0) { ids[n++] = special; continue; }
        char piece[5] = {0};
        if (*p == ' ') { memcpy(piece, "\xE2\x96\x81", 3); p++; }
        else {
            piece[0] = *p++;
            if ((piece[0] & 0xC0) == 0xC0)
                for (int i = 1; i < 4 && (*p & 0xC0) == 0x80; i++) piece[i] = *p++;
        }
        const VocabEntry *e = find_token(t, piece);
        if (e) { ids[n++] = e->id; continue; }

        for (char *b = piece; *b; b++) {
            char fb[8];
            snprintf(fb, sizeof fb, "<0x%02X>", (unsigned char)*b);
            e = find_token(t, fb);
            if (n >= cap || !e) return -1;
            ids[n++] = e->id;
        }
    }

    n = 1 + bpe(t, ids + 1, n - 1);
    ids[0] = 2;
    return n;
}

const char *decode(const Tokenizer *t, int token) {
    // Exporter-built table: specials are empty, <0xHH> is raw, ▁ is space, and other pieces are UTF-8.
    return token >= 0 && token < VOCAB_MAX ? t->decoded_tokens[token] : "";
}

// ----------------------------------------------------------------------------
// Kernels (AVX2 with a localized VNNI dot-product acceleration)

void quantize(int8_t *q, float *qs, const float *x, size_t rows, size_t k) {
    #pragma omp for schedule(static)
    for (size_t t = 0; t < rows * (k / 64); t++) {
        const float *g = x + t * 64;
        float amax = 0.0f;
        for (int j = 0; j < 64; j++) {
            float a = fabsf(g[j]);
            if (a > amax) amax = a;
        }
        float step = amax / 127.0f;
        float inv = step > 0.0f ? 1.0f / step : 0.0f;
        for (int j = 0; j < 64; j++) q[t * 64 + j] = (int8_t)rintf(g[j] * inv);
        qs[t] = step;
    }
}

static inline int threads(void) {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0xB, 0, eax, ebx, ecx, edx);
#ifdef _WIN32
    int logical = omp_get_num_procs();
#else
    int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    int smt = ebx & 0xffff;
    return getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : logical / (smt ? smt : 1);
}

void matmul_int8(float *c, const int8_t *a, const float *as, const Tensor *weight, size_t m) {
    size_t ng = (size_t)weight->shape[1] / 64;
    #pragma omp for schedule(static)
    for (size_t jb = 0; jb < (size_t)weight->shape[0] / PACKED_ROWS; jb++) {
        const int8_t *wb = (const int8_t *)weight->data + jb * PACKED_ROWS * weight->shape[1];
#if defined(__AVX512VNNI__)
        for (size_t i = 0; i < m; i += 8) {
            size_t rows = m - i < 8 ? m - i : 8;
            __m512 f[8] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                           _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
            for (size_t g = 0; g < ng; g++) {
                __m512i acc[8] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(),
                                  _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
                __m512i correction = _mm512_setzero_si512();
                __m512i offset_bytes = _mm512_set1_epi8(-128);
                for (int ch = 0; ch < 16; ch++) {
                    __m512i wv = _mm512_load_si512((const __m512i *)(wb + g * 1024 + ch * 64));
                    correction = _mm512_dpbusd_epi32(correction, offset_bytes, wv);
                    for (size_t q = 0; q < rows; q++) {
                        __m512i a8 = _mm512_broadcastd_epi32(
                            _mm_loadu_si32(a + (i + q) * weight->shape[1] + g * 64 + ch * 4));
                        a8 = _mm512_xor_si512(a8, offset_bytes);
                        acc[q] = _mm512_dpbusd_epi32(acc[q], a8, wv);
                    }
                }
                __m512 wsv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(weight->scales + (jb * ng + g) * PACKED_ROWS)));
                for (size_t q = 0; q < rows; q++)
                    f[q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[q], correction)), _mm512_mul_ps(wsv, _mm512_set1_ps(as[(i + q) * ng + g])), f[q]);
            }
            for (size_t q = 0; q < rows; q++)
                _mm512_storeu_ps(c + (i + q) * weight->shape[0] + jb * PACKED_ROWS, f[q]);
        }
#else
        for (size_t i = 0; i < m; i += 4) {
            size_t rows = m - i < 4 ? m - i : 4;
            for (int half = 0; half < 2; half++) {
                __m256 f[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                for (size_t g = 0; g < ng; g++) {
                    __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
                    for (int ch = 0; ch < 16; ch++) {
                        __m256i wv = _mm256_loadu_si256((const __m256i *)(wb + g * 1024 + ch * 64 + half * 32));
                        __m256i wa = _mm256_abs_epi8(wv);
                        for (size_t q = 0; q < rows; q++) {
                            __m256i a8 = _mm256_broadcastd_epi32(
                                _mm_loadu_si32(a + (i + q) * weight->shape[1] + g * 64 + ch * 4));
                            acc[q] = _mm256_add_epi32(acc[q], _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(a8, wv)), _mm256_set1_epi16(1)));
                        }
                    }
                    __m256 wsv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(weight->scales + (jb * ng + g) * PACKED_ROWS + half * 8)));
                    for (size_t q = 0; q < rows; q++)
                        f[q] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[q]), _mm256_mul_ps(wsv, _mm256_set1_ps(as[(i + q) * ng + g])), f[q]);
                }
            for (size_t q = 0; q < rows; q++)
                _mm256_storeu_ps(c + (i + q) * weight->shape[0] + jb * PACKED_ROWS + half * 8, f[q]);
            }
        }
#endif
    }
}

void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu) {
    const float *table = (const float *)gelu->data;
    const int ntab = gelu->shape[0];
    const float lo = (float)gelu->shape[1];
    const float hi = (float)gelu->shape[2];
    const float scale = (float)(ntab - 1) / (hi - lo);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lo) {
                x = table[0];
            } else if (!(x >= hi)) {
                float t = (x - lo) * scale;
                int j = (int)t;
                float w = t - (float)j;
                x = table[j] + w * (table[j + 1] - table[j]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}

void qk(float *att, const float *q, const float *kbase, int keys, int head_dim, int kv_dim) {
    for (size_t t = 0; t < (size_t)keys; t++) {
        const float *kt = kbase + t * kv_dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (int j = 0; j < head_dim; j += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + j), _mm256_loadu_ps(kt + j), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + j + 8), _mm256_loadu_ps(kt + j + 8), a1);
        }
        __m256 v = _mm256_add_ps(a0, a1);
        __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        float sum = _mm_cvtss_f32(_mm_add_ss(s4, _mm_movehdup_ps(s4)));
        att[t] = sum;
    }
}

void av(float *out, const float *att, const float *v, int keys, int head_dim, int kv_stride) {
    for (int j = 0; j < head_dim; j += 64) {
        __m256 acc[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (size_t t = 0; t < (size_t)keys; t++) {
            const float *vt = v + t * kv_stride + j;
            __m256 weight = _mm256_set1_ps(att[t]);
            for (int u = 0; u < 8; u++)
                acc[u] = _mm256_fmadd_ps(weight, _mm256_loadu_ps(vt + u * 8), acc[u]);
        }
        for (int u = 0; u < 8; u++)
            _mm256_storeu_ps(out + j + u * 8, acc[u]);
    }
}

// ----------------------------------------------------------------------------
// Transformer Forward Pass

void embedding(float *out, const Tensor *embed, const int *tokens, size_t seq_len) {
    int width = embed->shape[1];
    int groups = width / 64;
    float embedding_scale = sqrtf((float)width);
    #pragma omp for schedule(static)
    for (size_t sidx = 0; sidx < seq_len; sidx++) {
        size_t block = (size_t)(tokens[sidx] / PACKED_ROWS);
        int row = tokens[sidx] % PACKED_ROWS;
        float *x = out + sidx * width;
        const int8_t *block_data = (const int8_t *)embed->data + block * PACKED_ROWS * width;
        const uint16_t *block_scales = embed->scales + block * groups * PACKED_ROWS;
        for (size_t g = 0; g < (size_t)groups; g++) {
            const int8_t *group = block_data + g * PACKED_ROWS * 64;
            float scale = _cvtsh_ss(block_scales[g * PACKED_ROWS + row]) * embedding_scale;
            for (int j = 0; j < 64; j++) {
                int chunk = j / 4;
                int offset = j % 4;
                x[g * 64 + j] = (float)group[chunk * PACKED_ROWS * 4 + row * 4 + offset] * scale;
            }
        }
    }
}

void per_layer_embedding(float *out, const Tensor *embed, const int *tokens, size_t seq_len, float multiplier) {
    int width = embed->shape[1];
    int groups = width / 64;
    float rsqrt2 = 1.0f / sqrtf(2.0f);

    #pragma omp for schedule(static)
    for (size_t sidx = 0; sidx < seq_len; sidx++) {
        const int8_t *q = (const int8_t *)embed->data + (size_t)tokens[sidx] * width;
        const uint16_t *scales = embed->scales + (size_t)tokens[sidx] * groups;
        float *row = out + sidx * width;
        for (int g = 0; g < groups; g++) {
            float group_scale = _cvtsh_ss(scales[g]) * multiplier;
            for (int j = 0; j < 64; j++) {
                int i = g * 64 + j;
                row[i] = (row[i] + (float)q[i] * group_scale) * rsqrt2;
            }
        }
    }
}

void rmsnorm(float *o, const float *x, const Tensor *weight, int size, float eps, size_t row_count) {
    const float *w = weight ? (const float *)weight->data : NULL;
    #pragma omp for schedule(static)
    for (size_t t = 0; t < row_count; t++) {
        const float *xt = x + t * size;
        float *ot = o + t * size;
        float ss = 0.0f;
        for (int i = 0; i < size; i++)
            ss += xt[i] * xt[i];
        ss = 1.0f / sqrtf(ss / (float)size + eps);
        for (int i = 0; i < size; i++)
            ot[i] = (w ? w[i] : 1.0f) * (ss * xt[i]);
    }
}

void softmax(float *x, int n) {
    float max = x[0];
    float sum = 1.0f;
    for (int i = 1; i < n; i++) {
        float value = x[i];
        if (value > max) {
            sum = sum * expf(max - value) + 1.0f;
            max = value;
        } else {
            sum += expf(value - max);
        }
    }
    for (int i = 0; i < n; i++) x[i] = expf(x[i] - max) / sum;
}

void accum(float *a, const float *b, size_t n) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] += b[i];
}

void rope(const Tensor *cos, const Tensor *sin, float *x, int n_heads, int head_dim, int start_pos, size_t seq_len) {
    int pairs = cos->shape[1];
    #pragma omp for schedule(static)
    for (size_t t = 0; t < seq_len; t++) {
        const float *c = (float *)cos->data + (start_pos + t) * pairs;
        const float *s = (float *)sin->data + (start_pos + t) * pairs;
        for (size_t h = 0; h < (size_t)n_heads; h++) {
            float *v = x + (t * n_heads + h) * head_dim;
            for (int j = 0; j < pairs; j++) {
                float a = v[j];
                float b = v[j + head_dim / 2];
                v[j] = a * c[j] - b * s[j];
                v[j + head_dim / 2] = b * c[j] + a * s[j];
            }
        }
    }
}

void attention(RunState *s, const Config *c, const Gemma4LayerWeights *w, int layer,
               int pos0, size_t seq_len, float *att) {
    int window = c->layer_types[layer] == 's' ? c->sliding_window : 0;
    int head_dim = w->q_norm.shape[0];
    int q_dim = w->q_proj.shape[0];
    int kv_dim = s->kv_dim[layer];
    float *key_cache = s->kv_cache[layer];
    float *value_cache = key_cache + (size_t)c->max_seq_len * kv_dim;

    quantize(s->xq, s->xq_scales, s->hidden, seq_len, w->q_proj.shape[1]);

    matmul_int8(s->branch, s->xq, s->xq_scales, &w->q_proj, seq_len);
    rmsnorm(s->branch, s->branch, &w->q_norm, head_dim, c->norm_eps, seq_len * (q_dim / head_dim));
    rope(&w->rope_cos, &w->rope_sin, s->branch, q_dim / head_dim, head_dim, pos0, seq_len);

    if (layer < NUM_KV_CACHE_OWNERS) {
        float *new_k = key_cache + (size_t)pos0 * kv_dim;
        float *new_v = value_cache + (size_t)pos0 * kv_dim;
        matmul_int8(new_k, s->xq, s->xq_scales, &w->k_proj, seq_len);
        matmul_int8(new_v, s->xq, s->xq_scales, &w->v_proj, seq_len);
        rmsnorm(new_k, new_k, &w->k_norm, head_dim, c->norm_eps, seq_len * (kv_dim / head_dim));
        rmsnorm(new_v, new_v, NULL, head_dim, c->norm_eps, seq_len * (kv_dim / head_dim));
        rope(&w->rope_cos, &w->rope_sin, new_k, kv_dim / head_dim, head_dim, pos0, seq_len);
    }

    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (size_t h = 0; h < (size_t)(q_dim / head_dim); h++) {
        for (size_t s0 = 0; s0 < seq_len; s0++) {
            int first_key = window && pos0 + (int)s0 + 1 > window ? pos0 + (int)s0 + 1 - window : 0;
            int num_keys = pos0 + (int)s0 + 1 - first_key;
            qk(att, s->branch + s0 * q_dim + h * head_dim, key_cache + first_key * kv_dim + (h / (q_dim / kv_dim)) * head_dim, num_keys, head_dim, kv_dim);
            softmax(att, num_keys);
            av(s->hidden + s0 * q_dim + h * head_dim, att, value_cache + first_key * kv_dim + (h / (q_dim / kv_dim)) * head_dim, num_keys, head_dim, kv_dim);
        }
    }

    quantize(s->xq, s->xq_scales, s->hidden, seq_len, w->o_proj.shape[1]);
    matmul_int8(s->hidden, s->xq, s->xq_scales, &w->o_proj, seq_len);
}

void mlp(RunState *s, const Gemma4LayerWeights *w, size_t seq_len, const Tensor *gelu) {
    int intermediate_size = w->gate_proj.shape[0];

    quantize(s->xq, s->xq_scales, s->hidden, seq_len, w->gate_proj.shape[1]);
    matmul_int8(s->hidden, s->xq, s->xq_scales, &w->gate_proj, seq_len);
    matmul_int8(s->branch, s->xq, s->xq_scales, &w->up_proj, seq_len);

    geglu(s->hidden, s->branch, seq_len, intermediate_size, intermediate_size, gelu);

    quantize(s->xq, s->xq_scales, s->hidden, seq_len, intermediate_size);
    matmul_int8(s->hidden, s->xq, s->xq_scales, &w->down_proj, seq_len);
}

void forward(Gemma4Model *m, RunState *s, const int *tokens, size_t seq_len, int start_pos) {
    #pragma omp parallel num_threads(threads())
    {
    float att[(size_t)start_pos + seq_len];
    embedding(s->residual, &m->weights.embed, tokens, seq_len);

    quantize(s->xq, s->xq_scales, s->residual, seq_len, HIDDEN_SIZE);
    matmul_int8(s->per_layer, s->xq, s->xq_scales, &m->weights.per_layer_model_projection, seq_len);
    // RMSNorm(x / sqrt(H), eps) equals RMSNorm(x, H * eps) in real arithmetic.
    rmsnorm(s->per_layer, s->per_layer, &m->weights.per_layer_projection_norm, m->config.hidden_size_per_layer_input, m->config.norm_eps * (float)HIDDEN_SIZE, seq_len * NUM_LAYERS);

    per_layer_embedding(s->per_layer, &m->weights.embed_per_layer, tokens, seq_len, sqrtf((float)m->config.hidden_size_per_layer_input));

    for (int l = 0; l < NUM_LAYERS; l++) {
        Gemma4LayerWeights *lw = &m->weights.layers[l];

        rmsnorm(s->hidden, s->residual, &lw->input_layernorm, HIDDEN_SIZE, m->config.norm_eps, seq_len);
        attention(s, &m->config, lw, l, start_pos, seq_len, att);
        rmsnorm(s->hidden, s->hidden, &lw->post_attn_layernorm, HIDDEN_SIZE, m->config.norm_eps, seq_len);
        accum(s->residual, s->hidden, seq_len * HIDDEN_SIZE);

        rmsnorm(s->hidden, s->residual, &lw->pre_ffn_layernorm, HIDDEN_SIZE, m->config.norm_eps, seq_len);
        mlp(s, lw, seq_len, &m->weights.gelu_table);
        rmsnorm(s->hidden, s->hidden, &lw->post_ffn_layernorm, HIDDEN_SIZE, m->config.norm_eps, seq_len);
        accum(s->residual, s->hidden, seq_len * HIDDEN_SIZE);

        quantize(s->xq, s->xq_scales, s->residual, seq_len, HIDDEN_SIZE);
        matmul_int8(s->hidden, s->xq, s->xq_scales, &lw->per_layer_input_gate, seq_len);

        geglu(s->hidden, s->per_layer + l * m->config.hidden_size_per_layer_input, seq_len, m->config.hidden_size_per_layer_input, NUM_LAYERS * m->config.hidden_size_per_layer_input, &m->weights.gelu_table);

        quantize(s->xq, s->xq_scales, s->hidden, seq_len, m->config.hidden_size_per_layer_input);
        matmul_int8(s->hidden, s->xq, s->xq_scales, &lw->per_layer_projection, seq_len);
        rmsnorm(s->hidden, s->hidden, &lw->post_per_layer_input_norm, HIDDEN_SIZE, m->config.norm_eps, seq_len);
        float layer_scale = ((float *)lw->layer_scalar.data)[0];
        #pragma omp for schedule(static)
        for (size_t i = 0; i < seq_len * HIDDEN_SIZE; i++)
            s->residual[i] = (s->residual[i] + s->hidden[i]) * layer_scale;
    }
    }
}

void logits(Gemma4Model *m, RunState *s, size_t idx) {
    #pragma omp parallel num_threads(threads())
    {
        rmsnorm(s->hidden, s->residual + idx * HIDDEN_SIZE, &m->weights.norm, HIDDEN_SIZE, m->config.norm_eps, 1);
        quantize(s->xq, s->xq_scales, s->hidden, 1, HIDDEN_SIZE);
        matmul_int8(s->logits, s->xq, s->xq_scales, &m->weights.embed, 1);
        float cap = m->config.final_logit_softcapping;
        float inv = 1.0f / m->config.final_logit_softcapping;
        #pragma omp for schedule(static)
        for (int i = 0; i < VOCAB_MAX; i++) s->logits[i] = cap * tanhf(s->logits[i] * inv);
    }
}

// ----------------------------------------------------------------------------
// Sampler

unsigned long long rng_state = 42;

float random_f32(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int sample(float *logits, int n, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    struct { float score; int token; } top[64];
    for (int i = 0; i < 64; i++) top[i].score = -INFINITY;
    for (int token = 0; token < n; token++) {
        if (logits[token] <= top[63].score) continue;
        int i = 63;
        while (i > 0 && logits[token] > top[i - 1].score) { top[i] = top[i - 1]; i--; }
        top[i].score = logits[token]; top[i].token = token;
    }
    float sum = 0.0f, max = top[0].score / temperature;
    for (int i = 0; i < 64; i++) sum += top[i].score = expf(top[i].score / temperature - max);
    float mass = 0.0f;
    int count = 0;
    while (mass < 0.95f * sum) mass += top[count++].score;
    float r = random_f32() * mass;
    for (int i = 0; i < count; i++)
        if ((r -= top[i].score) <= 0.0f) return top[i].token;
    return top[count - 1].token;
}

// ----------------------------------------------------------------------------
// Generation

void prefill(Gemma4Model *m, RunState *s, const int *ids, int n, int dump) {
    for (int pos = 0; pos < n; pos += BATCH_SEQ) {
        int chunk = n - pos < BATCH_SEQ ? n - pos : BATCH_SEQ;
        forward(m, s, ids + pos, chunk, pos);
        if (dump) {
            for (int i = 0; i < chunk; i++) {
                logits(m, s, i);
                fwrite(s->logits, sizeof(float), VOCAB_MAX, stdout);
            }
        }
    }
}

char *wrap_prompt(const char *prompt) {
    const char *prefix = "<|turn>user\n", *suffix = "<turn|>\n<|turn>model\n";
    size_t size = strlen(prefix) + strlen(prompt) + strlen(suffix) + 1;
    char *wrapped = malloc(size);
    snprintf(wrapped, size, "%s%s%s", prefix, prompt, suffix);
    return wrapped;
}

void generate(Gemma4Model *m, RunState *s, const char *prompt, int max_tokens, float temperature, int dump) {
    Tokenizer *t = &m->tokenizer;
    int styled = !dump && isatty(STDOUT_FILENO);

    if (max_tokens > m->config.max_seq_len)
        max_tokens = m->config.max_seq_len;
    int *ids = malloc(max_tokens * sizeof(int));
    char *wrapped = dump ? NULL : wrap_prompt(prompt);
    int n = encode(t, wrapped ? wrapped : prompt, ids, max_tokens);
    if (n < 0) {
        fprintf(stderr, "prompt does not fit in %d tokens\n", max_tokens);
        free(wrapped);
        exit(1);
    }

    if (styled) {
        fputs("\n\033[2;36m────────────────────────────────\033[0m\n", stdout);
        fflush(stdout);
    }

    prefill(m, s, ids, n, dump);
    free(wrapped);
    if (dump) { free(ids); return; }

    for (int pos = n; pos < max_tokens; pos++) {
        logits(m, s, pos == n ? (n - 1) % BATCH_SEQ : 0);
        int next = sample(s->logits, VOCAB_MAX, temperature);
        if (next == 1 || next == 106) break;

        fputs(decode(t, next), stdout);
        fflush(stdout);
        forward(m, s, &next, 1, pos);
    }
    putchar('\n');
    free(ids);
}

// ----------------------------------------------------------------------------
// Bench

double now(void) {
#ifdef _WIN32
    static double frequency;
    LARGE_INTEGER ticks;
    if (!frequency) {
        QueryPerformanceFrequency(&ticks);
        frequency = (double)ticks.QuadPart;
    }
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart / frequency;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

void bench(Gemma4Model *m, RunState *s, int n_prompt, int n_gen) {
    if (n_prompt > 0) {
        int *ids = malloc(n_prompt * sizeof(int));
        for (int i = 0; i < n_prompt; i++)
            ids[i] = 2 + i % 1000;
        double start = now();
        prefill(m, s, ids, n_prompt, 0);
        printf("pp%d %.2f tok/s\n", n_prompt, (double)n_prompt / (now() - start));
        free(ids);
    }
    if (n_gen > 0) {
        int token = 2;
        double start = now();
        for (int pos = 0; pos < n_gen; pos++) {
            forward(m, s, &token, 1, pos);
            logits(m, s, 0);
            token = sample(s->logits, VOCAB_MAX, 0.0f);
        }
        printf("tg%d %.2f tok/s\n", n_gen, (double)n_gen / (now() - start));
    }
}

int main(int argc, char **argv) {
    const char *path = "gemma4-E2B-int8.mog";
    const char *prompt = "Why is the sky blue?";
    float temperature = 1.0f;
    int max_tokens = 1024;
    int do_bench = 0, dump = 0, n_pp = 0, n_tg = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) {
            do_bench = 1;
            if (i + 1 < argc) n_pp = atoi(argv[++i]);
            if (i + 1 < argc) n_tg = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--dump-logits")) dump = 1;
        else prompt = argv[i];
    }
#ifdef _WIN32
    if (dump) _setmode(_fileno(stdout), _O_BINARY);
#endif
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); return 1; }
    Gemma4Model *m = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }

    if (load_model(m) != 0) { fprintf(stderr, "bad mog file\n"); return 1; }
    RunState *s = calloc(1, sizeof(*s));
    init_run_state(s, &m->config, &m->weights);

    rng_state = (unsigned long long)(now() * 1e9);
    if (do_bench) bench(m, s, n_pp, n_tg);
    else generate(m, s, prompt, max_tokens, temperature, dump);
    free(s->residual);
    free(s);
    munmap(m, (size_t)st.st_size);
    return 0;
}

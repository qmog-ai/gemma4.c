#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <immintrin.h>

#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 1

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char key[8]; // Holds one UTF-8 piece or a pair of 32-bit token IDs.
    int result, rank;
} LookupEntry;

typedef struct {
    char token[94]; // The longest vocabulary piece is 93 bytes plus the null terminator.
    int id;
} VocabEntry;

typedef struct {
    int merge_count;
    int encode_vocab_count;
    int special_count;
    char decoded_tokens[VOCAB_SIZE][94];
    VocabEntry specials[256]; // Reserves space for the checkpoint's 24 special tokens.
    LookupEntry encode_vocab[32768]; // Reserves space for 19,249 directly encoded pieces.
    LookupEntry merges[514906]; // This checkpoint contains 514,906 merge rules.
} Tokenizer;

int compare_lookup_keys(const void *key, const void *entry) {
    return memcmp(key, ((const LookupEntry *)entry)->key, 8);
}

// Repeatedly applies the highest-priority learned merge until no adjacent token pair matches.
int apply_bpe_merges(const Tokenizer *tokenizer, int *tokens, int count) {
    for (;;) {
        const LookupEntry *best_merge = NULL;
        int position = -1;
        for (int i = 0; i + 1 < count; i++) {
            const LookupEntry *merge = bsearch(tokens + i, tokenizer->merges, tokenizer->merge_count, sizeof(LookupEntry), compare_lookup_keys);
            if (merge && (!best_merge || merge->rank < best_merge->rank)) { best_merge = merge; position = i; }
        }
        if (!best_merge) return count;

        tokens[position] = best_merge->result;
        memmove(tokens + position + 1, tokens + position + 2, (count - position - 2) * sizeof(*tokens));
        count--;
    }
}

// Converts the three prompt segments from UTF-8 into vocabulary pieces, falls back to byte tokens when needed, applies BPE, and prepends <bos>.
int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity) {
    int count = 1;
    for (int segment = 0; segment < 3; segment++)
        for (const char *cursor = segments[segment]; *cursor;) {
            if (count >= capacity) return -1;
            int special = -1;
            if (*cursor == '<')
                for (int i = 0; i < tokenizer->special_count && special < 0; i++) {
                    int length = (int)strlen(tokenizer->specials[i].token);
                    if (!strncmp(cursor, tokenizer->specials[i].token, length)) { special = tokenizer->specials[i].id; cursor += length; }
                }
            if (special >= 0) { tokens[count++] = special; continue; }
            char piece[8] = {0};
            if (*cursor == ' ') { memcpy(piece, "\xE2\x96\x81", 3); cursor++; } // SentencePiece represents spaces with U+2581.
            else {
                piece[0] = *cursor++;
                if ((piece[0] & 0xC0) == 0xC0)
                    for (int i = 1; i < 4 && (*cursor & 0xC0) == 0x80; i++) piece[i] = *cursor++;
            }
            const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab, tokenizer->encode_vocab_count,
                                      sizeof(LookupEntry), compare_lookup_keys);
            if (entry) { tokens[count++] = entry->result; continue; }

            for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
                if (count >= capacity) return -1;
                tokens[count++] = 238 + *byte; // Byte tokens occupy IDs 238 through 493.
            }
        }
    count = 1 + apply_bpe_merges(tokenizer, tokens + 1, count - 1);
    tokens[0] = 2; // Token 2 is <bos>.
    return count;
}

const char *token_text(const Tokenizer *tokenizer, int token) {
    return token >= 0 && token < VOCAB_SIZE ? tokenizer->decoded_tokens[token] : "";
}

// ----------------------------------------------------------------------------
// Model

// data and scales begin as file offsets and become pointers after the model is memory-mapped.
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
} LayerWeights;

typedef struct {
    Tensor embed;
    Tensor embed_per_layer;
    LayerWeights layers[NUM_LAYERS];
    Tensor norm;
    Tensor per_layer_model_projection;
    Tensor per_layer_projection_norm;
    Tensor gelu_table;
} ModelWeights;

typedef struct {
    float residual[BATCH_SIZE * HIDDEN_SIZE];                           // Carries each token's hidden state through all 35 layers.
    float hidden[VOCAB_SIZE];                                           // Reused for intermediate results and sized for the final vocabulary logits.
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];                      // Holds a second intermediate when attention or the MLP needs two results at once.
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];                     // Holds the current linear input after dynamic int8 quantization.
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];         // Stores one float scale for every 64 quantized values.
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];              // Stores one 256-value conditioning vector for every token and layer.
    float sliding_cache[3][4][2 * SLIDING_WINDOW * 256];                // Keeps keys and values for the previous 512 positions in twelve sliding caches.
    float full_cache[3][2 * MAX_CONTEXT * 512];                         // Holds the complete context for three 512-wide full-attention KV caches.
    int token_ids[MAX_CONTEXT];                                         // Holds the tokenized prompt before prefill.
} InferenceState;

typedef struct {
    char magic[4];
    Tokenizer tokenizer;
    ModelWeights weights;
} Model;

// Verifies that the compiler laid out the memory-mapped model exactly as the exporter expects.
_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8 && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96 && sizeof(LookupEntry) == 16 && sizeof(Tokenizer) == 33429932 && sizeof(Tensor) == 32 && sizeof(ModelWeights) == 21472 && sizeof(Model) == 33451408 && offsetof(Model, weights) == 33429936 && BATCH_SIZE == 1 && !(SLIDING_WINDOW & (SLIDING_WINDOW - 1)) && !(MAX_CONTEXT & (MAX_CONTEXT - 1)), "MOR ABI mismatch");

// ----------------------------------------------------------------------------
// Kernels

// Converts the model's IEEE 754 half-precision scales without CPU intrinsics.
float half_to_float(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 0x1f;
    uint32_t fraction = half & 0x03ff;
    uint32_t bits;

    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while (!(fraction & 0x0400)) {
                fraction <<= 1;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((fraction & 0x03ff) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (fraction << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Horizontally reduces eight int32 lanes.
static inline int32_t horizontal_sum_i32x8(__m256i values) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(values), _mm256_extracti128_si256(values, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

// Computes one signed int8 dot product. Lanes follow the input dimension;
// only one output row is active at a time.
#if defined(__AVX512VNNI__)
static inline int32_t dot64_row(const int8_t *input, const int8_t *weights) {
    __m512i dot = _mm512_setzero_si512();
    for (int k = 0; k < 64; k += 32) {
        __m512i input16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(input + k)));
        __m512i weight16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(weights + k)));
        dot = _mm512_dpwssd_epi32(dot, input16, weight16);
    }
    return _mm512_reduce_add_epi32(dot);
}
#else
static inline int32_t dot64_row(const int8_t *input, const int8_t *weights) {
    __m256i dot = _mm256_setzero_si256();
    for (int k = 0; k < 64; k += 16) {
        __m256i input16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(input + k)));
        __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(weights + k)));
        dot = _mm256_add_epi32(dot, _mm256_madd_epi16(input16, weight16));
    }
    return horizontal_sum_i32x8(dot);
}
#endif

static inline float scalar_weight_scale(const Tensor *weight, size_t output, size_t input_group) {
    size_t groups = (size_t)weight->shape[1] / 64;
    return half_to_float(weight->scales[output * groups + input_group]);
}

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    size_t outputs = (size_t)weight->shape[0];
    size_t inputs = (size_t)weight->shape[1];
    size_t groups = inputs / 64;
    const int8_t *weights = (const int8_t *)weight->data;
    #pragma omp for collapse(2) schedule(static)
    for (size_t row = 0; row < rows; row++) {
        for (size_t out = 0; out < outputs; out++) {
            float sum = 0.0f;
            for (size_t group = 0; group < groups; group++) {
                const int8_t *input_group = input_q + row * inputs + group * 64;
                const int8_t *weight_group = weights + out * inputs + group * 64;
                int32_t dot = dot64_row(input_group, weight_group);
                sum += (float)dot * input_scales[row * groups + group]
                     * scalar_weight_scale(weight, out, group);
            }
            output[row * outputs + out] = sum;
        }
    }
}


// Converts each input row to int8 in groups of 64 values with a float scale recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;
        float max_abs = 0.0f;
        for (int j = 0; j < 64; j++) {
            float value = fabsf(group[j]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < 64; j++) quantized[group_index * 64 + j] = (int8_t)rintf(group[j] * inverse_scale);
        scales[group_index] = scale;
    }
}

void attention_scores(float *scores, const float *query, const float *key_cache, int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) sum += query[j] * key[j];
        scores[key_index] = sum;
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache, int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int j = 0; j < head_dim; j++) {
        float sum = 0.0f;
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + ((first_key + key_index) & cache_mask) * head_dim + j;
            sum += probabilities[key_index] * *value;
        }
        output[j] = sum;
    }
}

// Approximates GELU from the exported lookup table and multiplies it by the up projection to produce the MLP's gated activation.
void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu_table) {
    const float *table = (const float *)gelu_table->data;
    const int table_size = gelu_table->shape[0];
    const float lower = (float)gelu_table->shape[1];
    const float upper = (float)gelu_table->shape[2];
    const float scale = (float)(table_size - 1) / (upper - lower);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) {
                x = table[0];
            } else if (!(x >= upper)) {
                float position = (x - lower) * scale;
                int index = (int)position;
                float fraction = position - (float)index;
                x = table[index] + fraction * (table[index + 1] - table[index]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Transformer

// Looks up row-major int8 embedding rows and dequantizes them directly.
void embedding(float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier) {
    int width = table->shape[1];
    int groups = width / 64;
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        float *vector = output + token * width;
        const int8_t *row = (const int8_t *)table->data + (size_t)tokens[token] * width;
        const uint16_t *scales = table->scales + (size_t)tokens[token] * groups;
        for (size_t group_index = 0; group_index < (size_t)groups; group_index++) {
            float scale = half_to_float(scales[group_index]) * multiplier;
            for (int j = 0; j < 64; j++)
                vector[group_index * 64 + j] = (float)row[group_index * 64 + j] * scale;
        }
    }
}

void rmsnorm(float *output, const float *input, const Tensor *weight, int width, float epsilon, size_t row_count) {
    const float *weights = weight ? (const float *)weight->data : NULL;
    #pragma omp for schedule(static)
    for (size_t row = 0; row < row_count; row++) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        float sum_squares = 0.0f;
        for (int i = 0; i < width; i++)
            sum_squares += input_row[i] * input_row[i];
        float inverse_rms = 1.0f / sqrtf(sum_squares / (float)width + epsilon);
        for (int i = 0; i < width; i++)
            output_row[i] = (weights ? weights[i] : 1.0f) * (inverse_rms * input_row[i]);
    }
}

void add_and_scale(float *output, const float *addend, size_t count, float scale) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < count; i++) output[i] = (output[i] + addend[i]) * scale;
}

// Rotates pairs of query or key channels using each position's sine and cosine values so attention can distinguish token order.
void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count) {
    int pairs = cosines->shape[1];
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        const float *cosine = (float *)cosines->data + (start_pos + token) * pairs;
        const float *sine = (float *)sines->data + (start_pos + token) * pairs;
        for (size_t head = 0; head < (size_t)num_heads; head++) {
            float *vector = vectors + (token * num_heads + head) * head_dim;
            for (int j = 0; j < pairs; j++) {
                float first = vector[j];
                float second = vector[j + head_dim / 2];
                vector[j] = first * cosine[j] - second * sine[j];
                vector[j + head_dim / 2] = second * cosine[j] + first * sine[j];
            }
        }
    }
}

void softmax(float *values, int count) {
    float max = values[0], sum = 1.0f;
    for (int i = 1; i < count; i++) {
        if (values[i] > max) { sum = sum * expf(max - values[i]) + 1.0f; max = values[i]; } // Rescale the sum when a new maximum appears so expf() stays in range.
        else sum += expf(values[i] - max);
    }

    for (int i = 0; i < count; i++) values[i] = expf(values[i] - max) / sum;
}

// Builds queries, updates the KV cache, and computes causal attention over 512 tokens or the full context while shared layers reuse the latest compatible cache.
void attention(InferenceState *state, const LayerWeights *layers, int layer,
               int start_pos, size_t token_count, float *scores) {
    const LayerWeights *weights = &layers[layer];
    int full_attention = layer % 5 == 4; // Every fifth layer uses full attention.
    int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW;
    int cache_mask = cache_len - 1; // Both cache lengths are powers of two, so masking wraps positions without division.
    int head_dim = weights->q_norm.shape[0];
    int query_width = weights->q_proj.shape[0];
    int cache_owner = layer;
    while (!layers[cache_owner].k_proj.data || (cache_owner % 5 == 4) != full_attention) cache_owner--; // Shared layers reuse the latest cache of the same attention type.
    float *key_cache = full_attention ? state->full_cache[cache_owner / 5] : state->sliding_cache[cache_owner / 5][cache_owner % 5];
    float *value_cache = key_cache + (size_t)cache_len * head_dim;

    // Build the queries for every token in the batch.
    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->q_proj.shape[1]);
    matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->q_proj, token_count);
    rmsnorm(state->auxiliary, state->auxiliary, &weights->q_norm, head_dim, 1e-6f, token_count * (query_width / head_dim));
    apply_rope(&weights->rope_cos, &weights->rope_sin, state->auxiliary, query_width / head_dim, head_dim, start_pos, token_count);

    // Compute keys and values and write them to the cache. Only the first 15 layers
    // have these weights, every other layer reads a cache an earlier layer filled.
    if (weights->k_proj.data) {
        float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * head_dim;
        float *new_values = value_cache + ((size_t)start_pos & cache_mask) * head_dim;
        matmul_int8(new_keys, state->quantized, state->activation_scales, &weights->k_proj, token_count);
        matmul_int8(new_values, state->quantized, state->activation_scales, &weights->v_proj, token_count);
        rmsnorm(new_keys, new_keys, &weights->k_norm, head_dim, 1e-6f, token_count);
        // Value vectors are normalized without a learned weight.
        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count);
        apply_rope(&weights->rope_cos, &weights->rope_sin, new_keys, 1, head_dim, start_pos, token_count);
    }

    // Each head scores its query against the visible keys and averages their values.
    // Sliding-window layers see the last 512 keys, full-attention layers see everything.
    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (size_t head = 0; head < (size_t)(query_width / head_dim); head++) {
        for (size_t token = 0; token < token_count; token++) {
            int first_key = !full_attention && start_pos + (int)token + 1 > SLIDING_WINDOW ? start_pos + (int)token + 1 - SLIDING_WINDOW : 0;
            int num_keys = start_pos + (int)token + 1 - first_key;
            float *head_output = state->hidden + token * query_width + head * head_dim;
            const float *query = state->auxiliary + token * query_width + head * head_dim;
            attention_scores(scores, query, key_cache, first_key, num_keys, cache_mask, head_dim);
            softmax(scores, num_keys);
            weighted_value_sum(head_output, scores, value_cache, first_key, num_keys, cache_mask, head_dim);
        }
    }

    // Merge the heads back to the residual width.
    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->o_proj.shape[1]);
    matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->o_proj, token_count);
}

void forward(Model *model, InferenceState *state, const int *tokens, size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    #pragma omp parallel
    {
    float scores[(size_t)start_pos + token_count];
    embedding(state->residual, &model->weights.embed, tokens, token_count, sqrtf((float)HIDDEN_SIZE));

    // Build the token-conditioned input that each transformer layer will receive.
    quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
    matmul_int8(state->per_layer_inputs, state->quantized, state->activation_scales, &model->weights.per_layer_model_projection, token_count);
    rmsnorm(state->per_layer_inputs, state->per_layer_inputs, &model->weights.per_layer_projection_norm, per_layer_width, 1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS);

    embedding(state->hidden, &model->weights.embed_per_layer, tokens, token_count, sqrtf((float)per_layer_width));
    add_and_scale(state->per_layer_inputs, state->hidden, token_count * NUM_LAYERS * per_layer_width, 1.0f / sqrtf(2.0f));

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *weights = &model->weights.layers[layer];

        // Attention, normalized and added back onto the residual stream.
        rmsnorm(state->hidden, state->residual, &weights->input_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        attention(state, model->weights.layers, layer, start_pos, token_count, scores);
        rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        // Feed-forward network, down(gelu(gate) * up), quantizing activations to int8 before each matmul.
        rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->gate_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->gate_proj, token_count);
        matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->up_proj, token_count);
        geglu(state->hidden, state->auxiliary, token_count, weights->gate_proj.shape[0], weights->gate_proj.shape[0], &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->down_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->down_proj, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        // This layer's per-layer embedding row, gated and added with a learned scale.
        quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_input_gate, token_count);
        geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width, token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->per_layer_projection.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_projection, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, ((float *)weights->layer_scalar.data)[0]);
    }
    }
}

// Reuses the embedding matrix to turn the final token representation into vocabulary logits, then applies Gemma's tanh soft cap.
float *logits(Model *model, InferenceState *state, size_t token) {
    #pragma omp parallel
    {
    rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE, &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1);
    quantize(state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE);
    matmul_int8(state->hidden, state->quantized, state->activation_scales, &model->weights.embed, 1);
    #pragma omp for schedule(static)
    for (int i = 0; i < VOCAB_SIZE; i++) state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
    }
    return state->hidden;
}

// ----------------------------------------------------------------------------
// Generation

int greedy(const float *scores) {
    int best = 0;
    for (int i = 1; i < VOCAB_SIZE; i++)
        if (scores[i] > scores[best]) best = i;
    return best;
}

void prefill(Model *model, InferenceState *state, const int *tokens, int token_count) {
    for (int position = 0; position < token_count; position++)
        forward(model, state, tokens + position, 1, position);
}

void generate(Model *model, InferenceState *state, const char *prompt) {
    Tokenizer *tokenizer = &model->tokenizer;
    const char *segments[3] = {"<|turn>user\n", prompt, "<turn|>\n<|turn>model\n"};
    int prompt_tokens = tokenize(tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (prompt_tokens < 0) {
        fprintf(stderr, "prompt exceeds the %d-token context limit\n", MAX_CONTEXT);
        exit(1);
    }

    prefill(model, state, state->token_ids, prompt_tokens);
    int end = prompt_tokens + 256 < MAX_CONTEXT ? prompt_tokens + 256 : MAX_CONTEXT;
    for (int position = prompt_tokens; position < end; position++) {
        int next_token = greedy(logits(model, state, 0));
        if (next_token == 1 || next_token == 106) break;

        fputs(token_text(tokenizer, next_token), stdout);
        fflush(stdout);
        forward(model, state, &next_token, 1, position);
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-row-major-int8.bin";
    const char *prompt = "Why is the sky blue?";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else prompt = argv[i];
    }

    int fd = open(model_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(model_path); return 1; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return 1; }
    if (memcmp(model->magic, "MOR", 4) != 0) {
        fprintf(stderr, "bad row-major model file\n");
        return 1;
    }

    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }

    InferenceState *state = calloc(1, sizeof(*state));
    generate(model, state, prompt);
    free(state);
    munmap(model, (size_t)st.st_size);
    return 0;
}
//      |\__/,|   (`\_
//    *.|o o  |*   ) )
//---(((---(((------------------
//|                            |
//|          gemma4.c          |
//|____________________________|

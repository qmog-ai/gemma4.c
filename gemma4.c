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
#include <time.h>
#include <unistd.h>

#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 512

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char key[8];
    int result, rank;
} LookupEntry;

typedef struct {
    char token[94];
    int id;
} VocabEntry;

typedef struct {
    int merge_count;
    int encode_vocab_count;
    int special_count;
    char decoded_tokens[VOCAB_SIZE][94];
    VocabEntry specials[256];
    LookupEntry encode_vocab[32768];
    LookupEntry merges[514906];
} Tokenizer;

int compare_lookup_keys(const void *key, const void *entry) {
    return memcmp(key, ((const LookupEntry *)entry)->key, 8);
}

int apply_bpe_merges(const Tokenizer *tokenizer, int *tokens, int count) {
    for (;;) {
        const LookupEntry *best_merge = NULL;
        int position = -1;
        for (int i = 0; i + 1 < count; i++) {
            const LookupEntry *merge = bsearch(tokens + i, tokenizer->merges, tokenizer->merge_count,
                                               sizeof(LookupEntry), compare_lookup_keys);
            if (merge && (!best_merge || merge->rank < best_merge->rank)) {
                best_merge = merge;
                position = i;
            }
        }
        if (!best_merge) return count;

        tokens[position] = best_merge->result;
        memmove(tokens + position + 1, tokens + position + 2,
                (count - position - 2) * sizeof(*tokens));
        count--;
    }
}

int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity) {
    int count = 1;

    for (int segment = 0; segment < 3; segment++) {
        for (const char *cursor = segments[segment]; *cursor;) {
            if (count >= capacity) return -1;
            int special = -1;
            if (*cursor == '<') {
                for (int i = 0; i < tokenizer->special_count && special < 0; i++) {
                    int length = (int)strlen(tokenizer->specials[i].token);
                    if (!strncmp(cursor, tokenizer->specials[i].token, length)) {
                        special = tokenizer->specials[i].id;
                        cursor += length;
                    }
                }
            }
            if (special >= 0) {
                tokens[count++] = special;
                continue;
            }

            char piece[8] = {0};
            if (*cursor == ' ') {
                memcpy(piece, "\xE2\x96\x81", 3);
                cursor++;
            } else {
                piece[0] = *cursor++;
                if ((piece[0] & 0xC0) == 0xC0) {
                    for (int i = 1; i < 4 && (*cursor & 0xC0) == 0x80; i++) piece[i] = *cursor++;
                }
            }

            const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab,
                                               tokenizer->encode_vocab_count,
                                               sizeof(LookupEntry), compare_lookup_keys);
            if (entry) {
                tokens[count++] = entry->result;
                continue;
            }
            for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
                if (count >= capacity) return -1;
                tokens[count++] = 238 + *byte;
            }
        }
    }

    count = 1 + apply_bpe_merges(tokenizer, tokens + 1, count - 1);
    tokens[0] = 2;
    return count;
}

const char *token_text(const Tokenizer *tokenizer, int token) {
    return token >= 0 && token < VOCAB_SIZE ? tokenizer->decoded_tokens[token] : "";
}

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
    float residual[BATCH_SIZE * HIDDEN_SIZE];
    float hidden[BATCH_SIZE * 8 * HIDDEN_SIZE];
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];
    float sliding_cache[3][4][2 * (SLIDING_WINDOW + BATCH_SIZE) * 256];
    float full_cache[3][2 * MAX_CONTEXT * 512];
    int token_ids[MAX_CONTEXT];
} InferenceState;

typedef struct {
    char magic[4];
    Tokenizer tokenizer;
    ModelWeights weights;
} Model;

_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8
               && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96
               && sizeof(LookupEntry) == 16 && sizeof(Tokenizer) == 33429932
               && sizeof(Tensor) == 32 && sizeof(ModelWeights) == 21472
               && sizeof(Model) == 33451408 && offsetof(Model, weights) == 33429936,
               "model ABI mismatch");

Model *load_model(const char *path, size_t *file_size) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        perror(path);
        return NULL;
    }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    if (memcmp(model->magic, "MOR", 4) != 0) {
        fprintf(stderr, "bad model file\n");
        munmap(model, (size_t)st.st_size);
        return NULL;
    }

    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data
            ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales
            ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    *file_size = (size_t)st.st_size;
    return model;
}

// ----------------------------------------------------------------------------
// Scalar kernels

float fp16_to_float(uint16_t value) {
    int sign = value >> 15;
    int exponent = (value >> 10) & 31;
    int fraction = value & 1023;
    float result;
    if (exponent == 0) result = ldexpf((float)fraction, -24);
    else if (exponent == 31) result = fraction ? NAN : INFINITY;
    else result = ldexpf((float)(1024 + fraction), exponent - 25);
    return sign ? -result : result;
}

float weight_scale(const Tensor *weight, int output, int input_group) {
    int groups = weight->shape[1] / 64;
    return fp16_to_float(weight->scales[(size_t)output * groups + input_group]);
}

void quantize(int8_t *output, float *scales, const float *input, size_t rows, int width) {
    for (size_t group = 0; group < rows * (size_t)(width / 64); group++) {
        float max_abs = 0.0f;
        for (int k = 0; k < 64; k++) {
            float value = fabsf(input[group * 64 + k]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int k = 0; k < 64; k++) {
            output[group * 64 + k] = (int8_t)rintf(input[group * 64 + k] * inverse_scale);
        }
        scales[group] = scale;
    }
}

void matmul_int8(float *output, const int8_t *input, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    int outputs = weight->shape[0];
    int inputs = weight->shape[1];
    int groups = inputs / 64;
    for (size_t row = 0; row < rows; row++) {
        for (int j = 0; j < outputs; j++) {
            float sum = 0.0f;
            for (int group = 0; group < groups; group++) {
                int dot = 0;
                for (int k = 0; k < 64; k++) {
                    int input_index = group * 64 + k;
                    const int8_t *values = weight->data;
                    dot += (int)input[row * inputs + input_index]
                           * (int)values[(size_t)j * inputs + input_index];
                }
                sum += (float)dot * input_scales[row * groups + group]
                       * weight_scale(weight, j, group);
            }
            output[row * outputs + j] = sum;
        }
    }
}

void embedding(float *output, const Tensor *table, const int *tokens,
               size_t token_count, float multiplier) {
    int width = table->shape[1];
    int groups = width / 64;
    for (size_t token_index = 0; token_index < token_count; token_index++) {
        float *vector = output + token_index * width;
        const int8_t *row = (const int8_t *)table->data + (size_t)tokens[token_index] * width;
        const uint16_t *scales = table->scales + (size_t)tokens[token_index] * groups;
        for (int group = 0; group < groups; group++) {
            const int8_t *values = row + group * 64;
            float scale = fp16_to_float(scales[group]) * multiplier;
            for (int j = 0; j < 64; j++) {
                vector[group * 64 + j] = (float)values[j] * scale;
            }
        }
    }
}

void rmsnorm(float *output, const float *input, const Tensor *weight,
             int width, float epsilon, size_t rows) {
    const float *weights = weight ? (const float *)weight->data : NULL;
    for (size_t row = 0; row < rows; row++) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        float sum_squares = 0.0f;
        for (int i = 0; i < width; i++) sum_squares += input_row[i] * input_row[i];
        float inverse_rms = 1.0f / sqrtf(sum_squares / (float)width + epsilon);
        for (int i = 0; i < width; i++) {
            output_row[i] = (weights ? weights[i] : 1.0f) * inverse_rms * input_row[i];
        }
    }
}

void add_and_scale(float *output, const float *addend, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) output[i] = (output[i] + addend[i]) * scale;
}

void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count) {
    int pairs = cosines->shape[1];
    for (size_t token = 0; token < token_count; token++) {
        const float *cosine = (float *)cosines->data + (start_pos + token) * pairs;
        const float *sine = (float *)sines->data + (start_pos + token) * pairs;
        for (int head = 0; head < num_heads; head++) {
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
    float max = values[0];
    for (int i = 1; i < count; i++) if (values[i] > max) max = values[i];
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += values[i] = expf(values[i] - max);
    for (int i = 0; i < count; i++) values[i] /= sum;
}

void geglu(float *gate, const float *up, int rows, int width, int up_stride,
           const Tensor *gelu_table) {
    const float *table = gelu_table->data;
    int table_size = gelu_table->shape[0];
    float lower = (float)gelu_table->shape[1];
    float upper = (float)gelu_table->shape[2];
    float scale = (float)(table_size - 1) / (upper - lower);
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) x = table[0];
            else if (x < upper) {
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

void attention_scores(float *scores, const float *query, const float *key_cache,
                      int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        float dot = 0.0f;
        for (int j = 0; j < head_dim; j++) dot += query[j] * key[j];
        scores[key_index] = dot;
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
                        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int j = 0; j < head_dim; j++) {
        float sum = 0.0f;
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache
                + ((first_key + key_index) & cache_mask) * head_dim;
            sum += probabilities[key_index] * value[j];
        }
        output[j] = sum;
    }
}

void attention(InferenceState *state, const LayerWeights *layers, int layer,
               int start_pos, size_t token_count, float *scores) {
    const LayerWeights *weights = &layers[layer];
    int full_attention = layer % 5 == 4;
    int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE;
    int cache_mask = cache_len - 1;
    int head_dim = weights->q_norm.shape[0];
    int query_width = weights->q_proj.shape[0];
    int cache_owner = layer;
    while (!layers[cache_owner].k_proj.data || (cache_owner % 5 == 4) != full_attention) cache_owner--;
    float *key_cache = full_attention ? state->full_cache[cache_owner / 5]
                                      : state->sliding_cache[cache_owner / 5][cache_owner % 5];
    float *value_cache = key_cache + (size_t)cache_len * head_dim;

    quantize(state->quantized, state->activation_scales, state->hidden, token_count,
             weights->q_proj.shape[1]);
    matmul_int8(state->auxiliary, state->quantized, state->activation_scales,
                &weights->q_proj, token_count);
    rmsnorm(state->auxiliary, state->auxiliary, &weights->q_norm, head_dim, 1e-6f,
            token_count * (query_width / head_dim));
    apply_rope(&weights->rope_cos, &weights->rope_sin, state->auxiliary,
               query_width / head_dim, head_dim, start_pos, token_count);

    if (weights->k_proj.data) {
        float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * head_dim;
        float *new_values = value_cache + ((size_t)start_pos & cache_mask) * head_dim;
        matmul_int8(new_keys, state->quantized, state->activation_scales,
                    &weights->k_proj, token_count);
        matmul_int8(new_values, state->quantized, state->activation_scales,
                    &weights->v_proj, token_count);
        rmsnorm(new_keys, new_keys, &weights->k_norm, head_dim, 1e-6f, token_count);
        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count);
        apply_rope(&weights->rope_cos, &weights->rope_sin, new_keys, 1, head_dim,
                   start_pos, token_count);
    }

    int num_heads = query_width / head_dim;
    for (int head = 0; head < num_heads; head++) {
        for (size_t token = 0; token < token_count; token++) {
            int position = start_pos + (int)token;
            int first_key = !full_attention && position + 1 > SLIDING_WINDOW
                ? position + 1 - SLIDING_WINDOW : 0;
            int num_keys = position + 1 - first_key;
            const float *query = state->auxiliary
                + (token * num_heads + head) * head_dim;
            float *head_output = state->hidden
                + token * query_width + head * head_dim;
            attention_scores(scores, query, key_cache, first_key, num_keys,
                             cache_mask, head_dim);
            softmax(scores, num_keys);
            weighted_value_sum(head_output, scores, value_cache, first_key,
                               num_keys, cache_mask, head_dim);
        }
    }
    quantize(state->quantized, state->activation_scales, state->hidden, token_count,
             weights->o_proj.shape[1]);
    matmul_int8(state->hidden, state->quantized, state->activation_scales,
                &weights->o_proj, token_count);
}

void forward(Model *model, InferenceState *state, const int *tokens,
             size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    float scores[(size_t)start_pos + token_count];

    embedding(state->residual, &model->weights.embed, tokens, token_count,
              sqrtf((float)HIDDEN_SIZE));
    quantize(state->quantized, state->activation_scales, state->residual,
             token_count, HIDDEN_SIZE);
    matmul_int8(state->per_layer_inputs, state->quantized, state->activation_scales,
                &model->weights.per_layer_model_projection, token_count);
    rmsnorm(state->per_layer_inputs, state->per_layer_inputs,
            &model->weights.per_layer_projection_norm, per_layer_width,
            1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS);
    embedding(state->hidden, &model->weights.embed_per_layer, tokens, token_count,
              sqrtf((float)per_layer_width));
    add_and_scale(state->per_layer_inputs, state->hidden,
                  token_count * NUM_LAYERS * per_layer_width, 1.0f / sqrtf(2.0f));

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *weights = &model->weights.layers[layer];

        rmsnorm(state->hidden, state->residual, &weights->input_layernorm,
                HIDDEN_SIZE, 1e-6f, token_count);
        attention(state, model->weights.layers, layer, start_pos, token_count, scores);
        rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm,
                HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm,
                HIDDEN_SIZE, 1e-6f, token_count);
        quantize(state->quantized, state->activation_scales, state->hidden,
                 token_count, weights->gate_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales,
                    &weights->gate_proj, token_count);
        matmul_int8(state->auxiliary, state->quantized, state->activation_scales,
                    &weights->up_proj, token_count);
        geglu(state->hidden, state->auxiliary, (int)token_count,
              weights->gate_proj.shape[0], weights->gate_proj.shape[0],
              &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden,
                 token_count, weights->down_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales,
                    &weights->down_proj, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm,
                HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        quantize(state->quantized, state->activation_scales, state->residual,
                 token_count, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales,
                    &weights->per_layer_input_gate, token_count);
        geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width,
              (int)token_count, per_layer_width, NUM_LAYERS * per_layer_width,
              &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden,
                 token_count, weights->per_layer_projection.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales,
                    &weights->per_layer_projection, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm,
                HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE,
                      ((float *)weights->layer_scalar.data)[0]);
    }
}

float *logits(Model *model, InferenceState *state, size_t token) {
    rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE,
            &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1);
    quantize(state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE);
    matmul_int8(state->hidden, state->quantized, state->activation_scales,
                &model->weights.embed, 1);
    for (int i = 0; i < VOCAB_SIZE; i++) {
        state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
    }
    return state->hidden;
}

void prefill(Model *model, InferenceState *state, const int *tokens, int token_count) {
    for (int position = 0; position < token_count; position++) {
        int chunk = 1;
        forward(model, state, tokens + position, (size_t)chunk, position);
    }
}

int greedy(float *scores) {
    int best = 0;
    for (int i = 1; i < VOCAB_SIZE; i++) if (scores[i] > scores[best]) best = i;
    return best;
}

void generate(Model *model, InferenceState *state, const char *prompt, int max_new_tokens) {
    const char *segments[3] = {"", prompt, ""};
    int count = tokenize(&model->tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (count < 0) {
        fprintf(stderr, "prompt is too long\n");
        exit(1);
    }
    prefill(model, state, state->token_ids, count);
    for (int position = count; position < count + max_new_tokens && position < MAX_CONTEXT; position++) {
        size_t row = 0;
        int token = greedy(logits(model, state, row));
        if (token == 1 || token == 106) break;
        fputs(token_text(&model->tokenizer, token), stdout);
        fflush(stdout);
        forward(model, state, &token, 1, position);
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    const char *prompt = "Why is the sky blue?";
    int max_new_tokens = 32;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else prompt = argv[i];
    }

    size_t file_size;
    Model *model = load_model(model_path, &file_size);
    if (!model) return 1;

    InferenceState *state = calloc(1, sizeof(*state));
    generate(model, state, prompt, max_new_tokens);
    free(state);
    munmap(model, file_size);
    return 0;
}

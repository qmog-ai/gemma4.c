#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072

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

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    const char *prompt = "Why is the sky blue?";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else prompt = argv[i];
    }

    size_t file_size;
    Model *model = load_model(model_path, &file_size);
    if (!model) return 1;

    int *tokens = malloc(MAX_CONTEXT * sizeof(*tokens));
    const char *segments[3] = {"", prompt, ""};
    int count = tokenize(&model->tokenizer, segments, tokens, MAX_CONTEXT);
    if (count < 0) {
        fprintf(stderr, "prompt is too long\n");
        return 1;
    }
    printf("%d tokens:", count);
    for (int i = 0; i < count; i++) printf(" %d", tokens[i]);
    putchar('\n');

    free(tokens);
    munmap(model, file_size);
    return 0;
}

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

// ----------------------------------------------------------------------------
// Model

typedef struct {
    void *data;
    float *scales;
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
    if (memcmp(model->magic, "MOF", 4) != 0) {
        fprintf(stderr, "bad model file\n");
        munmap(model, (size_t)st.st_size);
        return NULL;
    }

    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data
            ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales
            ? (float *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    *file_size = (size_t)st.st_size;
    return model;
}

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
    }

    size_t file_size;
    Model *model = load_model(model_path, &file_size);
    if (!model) return 1;
    munmap(model, file_size);
    return 0;
}

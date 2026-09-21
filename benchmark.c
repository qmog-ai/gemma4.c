/*
Benchmarks prompt processing (pp) or token generation (tg) without
tokenization or sampling. The model is loaded and --d tokens are written to
the KV cache before timing begins. Exactly one of --pp and --tg must be
nonzero.

Build and run:

  make benchmark
  ./benchmark -m gemma4-E2B-int8.bin --pp 512 --tg 0 --d 0
  ./benchmark -m gemma4-E2B-int8.bin --pp 0 --tg 128 --d 2048

Profile only the measured pp or tg work without function inlining:

  make profile
  perf record -m 64 -e cycles:pp -- ./benchmark -m gemma4-E2B-int8.bin --pp 0 --tg 256 --d 0
  perf report
*/

#define GEMMA4_NO_MAIN
#include "gemma4.c"

#include <linux/prctl.h>
#include <sys/prctl.h>
#include <time.h>

static void perf_events(int command) {
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    prctl(command);
}

int main(int argc, char **argv) {
    prctl(PR_TASK_PERF_EVENTS_DISABLE);
    const char *model_path = "gemma4-E2B-int8.bin";
    int pp = 0, tg = 0, depth = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--pp") && i + 1 < argc) pp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tg") && i + 1 < argc) tg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d") && i + 1 < argc) depth = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-m MODEL] --pp N --tg N --d N\n", argv[0]);
            return 1;
        }
    }
    if (pp < 0 || tg < 0 || (pp > 0) == (tg > 0) || depth < 0 ||
        depth + (pp ? pp : tg) > MAX_CONTEXT) {
        fprintf(stderr, "select either --pp or --tg within the %d-token context\n", MAX_CONTEXT);
        return 1;
    }

    size_t model_size;
    Model *model = load_model(model_path, &model_size);
    InferenceState *state = calloc(1, sizeof(*state));
    for (int i = 0; i < depth + pp; i++) state->token_ids[i] = 2 + i % 1000;

    // Establish the requested starting context before timing or profiling.
    prefill(model, state, state->token_ids, depth, 0, 0);

    perf_events(PR_TASK_PERF_EVENTS_ENABLE);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (pp) {
        int last_row = prefill(model, state, state->token_ids + depth, pp, depth, 0);
        (void)logits(model, state, last_row);
    } else {
        const int token = 2;
        for (int position = depth; position < depth + tg; position++) {
            forward(model, state, &token, 1, position);
            (void)logits(model, state, 0);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    perf_events(PR_TASK_PERF_EVENTS_DISABLE);

    int tokens = pp ? pp : tg;
    printf("%s%d@d%d %.2f tok/s\n", pp ? "pp" : "tg", tokens, depth, tokens / elapsed);
    free(state);
    munmap(model, model_size);
    return 0;
}

/*
Benchmarks prompt processing (pp) or token generation (tg) without
tokenization or sampling. Each workload is warmed up once, then repeated.

  make benchmark
  ./benchmark -m gemma4-E2B-int8.bin --pp 64,256,512 -r 3
  ./benchmark -m gemma4-E2B-int8.bin --tg 128 --d 64,256,512 -r 3

Build with function inlining disabled for profiling:

  ./profile.sh -m gemma4-E2B-int8.bin --tg 128 --d 512 -r 1
  perf report
*/

#define GEMMA4_NO_MAIN
#include "gemma4.c"

#include <errno.h>
#include <limits.h>
#include <time.h>

#define MAX_CASES 16

static int parse_list(const char *text, int *values) {
    int count = 0;
    while (*text) {
        char *end;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value > INT_MAX || count == MAX_CASES) return -1;
        values[count++] = (int)value;
        if (!*end) return count;
        if (*end != ',' || !end[1]) return -1;
        text = end + 1;
    }
    return -1;
}

static void workload(Model *model, InferenceState *state, int pp, int tg, int depth) {
    if (pp) {
        int last_row = prefill(model, state, state->token_ids + depth, pp, depth, 0);
        (void)logits(model, state, last_row);
    }

    const int token = 2;
    for (int position = depth + pp; position < depth + pp + tg; position++) {
        forward(model, state, &token, 1, position);
        (void)logits(model, state, 0);
    }
}

static void perf_command(int control_fd, int ack_fd, const char *command) {
    if (control_fd < 0) return;
    dprintf(control_fd, "%s\n", command);

    char ack[sizeof("ack\n")];
    ssize_t bytes;
    do bytes = read(ack_fd, ack, sizeof(ack)); while (bytes < 0 && errno == EINTR);
    if (bytes != sizeof(ack) || memcmp(ack, "ack\n", sizeof(ack))) {
        fprintf(stderr, "perf control failed\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    int prompt_lengths[MAX_CASES], depths[MAX_CASES];
    int prompt_count = 0, depth_count = 0, tg = 0, repetitions = 5;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--pp") && i + 1 < argc) prompt_count = parse_list(argv[++i], prompt_lengths);
        else if (!strcmp(argv[i], "--tg") && i + 1 < argc) tg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d") && i + 1 < argc) depth_count = parse_list(argv[++i], depths);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) repetitions = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-m MODEL] (--pp N,... | --tg N [--d N,...]) [-r N]\n", argv[0]);
            return 1;
        }
    }

    if (!depth_count) depths[depth_count++] = 0;
    if (prompt_count < 0 || depth_count < 0 || tg < 0 || repetitions < 1 ||
        (prompt_count > 0) == (tg > 0) || (prompt_count && depth_count > 1)) {
        fprintf(stderr, "select either --pp or --tg within the %d-token context\n", MAX_CONTEXT);
        return 1;
    }

    int case_count = prompt_count ? prompt_count : depth_count;
    for (int i = 0; i < case_count; i++) {
        int pp = prompt_count ? prompt_lengths[i] : 0;
        int depth = prompt_count ? depths[0] : depths[i];
        long long context = (long long)depth + pp + tg;
        if (!(pp + tg) || context > MAX_CONTEXT) {
            fprintf(stderr, "workload exceeds the %d-token context\n", MAX_CONTEXT);
            return 1;
        }
    }

    size_t model_size;
    Model *model = load_model(model_path, &model_size);
    double means[MAX_CASES], stddevs[MAX_CASES];
    const char *control = getenv("PERF_CTL_FD");
    const char *ack = getenv("PERF_ACK_FD");
    int perf_control_fd = control ? atoi(control) : -1;
    int perf_ack_fd = ack ? atoi(ack) : -1;

    for (int i = 0; i < case_count; i++) {
        int pp = prompt_count ? prompt_lengths[i] : 0;
        int depth = prompt_count ? depths[0] : depths[i];

        InferenceState *state = calloc(1, sizeof(*state));
        for (int token = 0; token < depth + pp; token++) state->token_ids[token] = 2 + token % 1000;
        prefill(model, state, state->token_ids, depth, 0, 0);

        // Full-cache prefixes are never overwritten, so only the sliding ring must be restored between runs.
        void *saved_sliding = malloc(sizeof(state->sliding_cache));
        memcpy(saved_sliding, state->sliding_cache, sizeof(state->sliding_cache));

        workload(model, state, pp, tg, depth);

        double sum = 0.0, sum_squared = 0.0;
        for (int repetition = 0; repetition < repetitions; repetition++) {
            memcpy(state->sliding_cache, saved_sliding, sizeof(state->sliding_cache));

            perf_command(perf_control_fd, perf_ack_fd, "enable");
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            workload(model, state, pp, tg, depth);
            clock_gettime(CLOCK_MONOTONIC, &end);
            perf_command(perf_control_fd, perf_ack_fd, "disable");

            double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
            double rate = (pp + tg) / elapsed;
            sum += rate;
            sum_squared += rate * rate;
        }

        free(saved_sliding);
        free(state);

        means[i] = sum / repetitions;
        double variance = repetitions > 1
            ? (sum_squared - repetitions * means[i] * means[i]) / (repetitions - 1)
            : 0.0;
        stddevs[i] = sqrt(fmax(variance, 0.0));
    }

    for (int i = 0; i < case_count; i++) {
        int pp = prompt_count ? prompt_lengths[i] : 0;
        int depth = prompt_count ? depths[0] : depths[i];
        printf("%s%d@d%d %.2f +/- %.2f tok/s\n",
               pp ? "pp" : "tg", pp ? pp : tg, depth, means[i], stddevs[i]);
    }

    munmap(model, model_size);
    return 0;
}

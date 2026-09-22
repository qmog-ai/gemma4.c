#!/usr/bin/env bash
set -euo pipefail

# Compile and benchmark prefill with batch sizes from 1 to 512.
# Each result is the median of three runs after one discarded warmup.
#
# Usage:
#   ./benchmark_batch_sizes.sh [model] [prompt_tokens] [output.csv]

root=$(cd "$(dirname "$0")" && pwd)
model=${1:-"$root/gemma4-E2B-int8.bin"}
prompt_tokens=${2:-512}
output=${3:-"$root/batch-size-results.csv"}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

printf 'batch_size,run_1,run_2,run_3,median_tok_s\n' > "$output"

for batch_size in 1 2 4 8 16 32 64 128 256 512; do
    binary="$build_dir/benchmark-$batch_size"
    cc -std=c11 -O3 -Wall -Wextra -march=native -fopenmp \
        -DBATCH_SIZE="$batch_size" "$root/benchmark.c" -o "$binary" -lm

    echo "batch size $batch_size" >&2
    "$binary" -m "$model" --pp "$prompt_tokens" --tg 0 --d 0 >/dev/null

    rates=()
    for run in 1 2 3; do
        rate=$("$binary" -m "$model" --pp "$prompt_tokens" --tg 0 --d 0 | awk '{print $2}')
        rates+=("$rate")
    done

    median=$(printf '%s\n' "${rates[@]}" | sort -n | sed -n '2p')
    printf '%s,%s,%s,%s,%s\n' \
        "$batch_size" "${rates[0]}" "${rates[1]}" "${rates[2]}" "$median" \
        | tee -a "$output"
done

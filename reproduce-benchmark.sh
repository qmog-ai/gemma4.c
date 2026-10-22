#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export OMP_NUM_THREADS=8

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
gemma_bin=${GEMMA_BIN:-"$repo_dir/run"}
gemma_model=${GEMMA_MODEL:-"$repo_dir/gemma4-E2B-int8.bin"}
llama_dir=${LLAMA_DIR:-/home/ryan/Projects/llama.cpp-q8-repack-test}
llama_bin=${LLAMA_BIN:-"$llama_dir/build/bin/llama-bench"}
g64_model=${G64_MODEL:-/home/ryan/Projects/models/gemma4/gemma-4-E2B-it-Q8_0-G64-final.gguf}
g32_model=${G32_MODEL:-/home/ryan/Projects/models/gemma4/gemma-4-E2B-it-Q8_0.gguf}
repetitions=${REPETITIONS:-3}
decode_tokens=${DECODE_TOKENS:-128}
read -r -a contexts <<< "${CONTEXTS:-512 2048 8192 16384}"
engines=(gemma g64 g32)

if [[ -n ${RESULT_DIR:-} ]]; then
    result_dir=$RESULT_DIR
else
    result_dir="$repo_dir/benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$result_dir"

raw_file="$result_dir/raw.tsv"
summary_file="$result_dir/summary.tsv"
transcript_file="$result_dir/transcript.log"
stderr_file="$result_dir/stderr.log"

for path in "$gemma_model" "$llama_bin" "$g64_model" "$g32_model"; do
    if [[ ! -e $path ]]; then
        printf 'missing required file: %s\n' "$path" >&2
        exit 1
    fi
done

if [[ ${BUILD:-1} == 1 ]]; then
    make -C "$repo_dir"
    cmake --build "$llama_dir/build" --target llama-bench -j "$(nproc)"
fi

if [[ ! -f $raw_file ]]; then
    printf 'context\trepetition\torder\tengine\tprefill_tps\tdecode_tps\ttimestamp_utc\n' > "$raw_file"
fi

{
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host\t%s\n' "$(uname -n)"
    printf 'kernel\t%s\n' "$(uname -srmo)"
    printf 'cpu\t%s\n' "$(lscpu | awk -F: '/Model name/ { sub(/^[ \t]+/, "", $2); print $2; exit }')"
    printf 'gemma_commit\t%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    printf 'llama_commit\t%s\n' "$(git -C "$llama_dir" rev-parse HEAD)"
    printf 'threads\t8\n'
    printf 'batch\t512\n'
    printf 'ubatch\t512\n'
    printf 'kv_cache\tf32\n'
    printf 'decode_tokens\t%s\n' "$decode_tokens"
    printf 'repetitions\t%s\n' "$repetitions"
    stat -c 'artifact\t%n\t%s bytes' "$gemma_model" "$g64_model" "$g32_model"
} > "$result_dir/metadata.tsv"
git -C "$repo_dir" diff -- gemma4.c > "$result_dir/gemma4.diff"
git -C "$llama_dir" diff -- tools/llama-bench/llama-bench.cpp > "$result_dir/llama-bench.diff"

parse_llama_csv() {
    awk -F, 'NF > 2 { value = $(NF - 1) } END { gsub(/"/, "", value); print value }'
}

run_gemma() {
    local context=$1 output prefill decode
    output=$("$gemma_bin" -m "$gemma_model" --bench "$context" "$decode_tokens" 2>>"$stderr_file")
    {
        printf '\n[gemma context=%s]\n' "$context"
        printf '%s\n' "$output"
    } >> "$transcript_file"
    prefill=$(awk '$1 ~ /^pp/ { print $2 }' <<< "$output")
    decode=$(awk '$1 ~ /^tg/ { print $2 }' <<< "$output")
    [[ -n $prefill && -n $decode ]]
    printf '%s\t%s\n' "$prefill" "$decode"
}

run_llama_test() {
    local model=$1 context=$2 phase=$3 output
    local args=(
        -m "$model" -t 8 -b 512 -ub 512 -ctk f32 -ctv f32 -ngl 0
        -r 1 --no-warmup -o csv
    )
    if [[ $phase == prefill ]]; then
        args+=(-p "$context" -n 0 -d 0)
    else
        args+=(-p 0 -n "$decode_tokens" -d "$context")
    fi
    output=$("$llama_bin" "${args[@]}" 2>>"$stderr_file")
    {
        printf '\n[llama model=%s context=%s phase=%s]\n' "$(basename "$model")" "$context" "$phase"
        printf '%s\n' "$output"
    } >> "$transcript_file"
    printf '%s\n' "$output" | parse_llama_csv
}

run_engine() {
    local engine=$1 context=$2 model prefill decode
    if [[ $engine == gemma ]]; then
        run_gemma "$context"
        return
    fi
    if [[ $engine == g64 ]]; then
        model=$g64_model
    else
        model=$g32_model
    fi
    prefill=$(run_llama_test "$model" "$context" prefill)
    decode=$(run_llama_test "$model" "$context" decode)
    [[ -n $prefill && -n $decode ]]
    printf '%s\t%s\n' "$prefill" "$decode"
}

already_measured() {
    local context=$1 repetition=$2 engine=$3
    awk -F '\t' -v c="$context" -v r="$repetition" -v e="$engine" \
        'NR > 1 && $1 == c && $2 == r && $4 == e { found = 1 } END { exit !found }' "$raw_file"
}

for context_index in "${!contexts[@]}"; do
    context=${contexts[$context_index]}
    printf '\ncontext %s: discarded warmups\n' "$context"
    warmup_start=$((context_index % ${#engines[@]}))
    for ((offset = 0; offset < ${#engines[@]}; offset++)); do
        engine=${engines[$(((warmup_start + offset) % ${#engines[@]}))]}
        marker="$result_dir/warmup-${context}-${engine}.done"
        if [[ ! -e $marker ]]; then
            printf '  warmup %s\n' "$engine"
            run_engine "$engine" "$context" > /dev/null
            touch "$marker"
        fi
    done

    for ((repetition = 1; repetition <= repetitions; repetition++)); do
        start=$(((context_index + repetition - 1) % ${#engines[@]}))
        for ((offset = 0; offset < ${#engines[@]}; offset++)); do
            order=$((offset + 1))
            engine=${engines[$(((start + offset) % ${#engines[@]}))]}
            if already_measured "$context" "$repetition" "$engine"; then
                printf '  skip context=%s repetition=%s engine=%s\n' "$context" "$repetition" "$engine"
                continue
            fi
            printf '  run context=%s repetition=%s order=%s engine=%s\n' "$context" "$repetition" "$order" "$engine"
            IFS=$'\t' read -r prefill decode < <(run_engine "$engine" "$context")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$context" "$repetition" "$order" "$engine" "$prefill" "$decode" \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$raw_file"
        done
    done
done

printf 'context\tengine\tn\tprefill_mean\tprefill_sd\tdecode_mean\tdecode_sd\n' > "$summary_file"
for context in "${contexts[@]}"; do
    for engine in "${engines[@]}"; do
        awk -F '\t' -v c="$context" -v e="$engine" '
            NR > 1 && $1 == c && $4 == e {
                n++; pp += $5; pp2 += $5 * $5; tg += $6; tg2 += $6 * $6
            }
            END {
                if (!n) exit 1
                pp_mean = pp / n; tg_mean = tg / n
                pp_sd = n > 1 ? sqrt((pp2 - pp * pp / n) / (n - 1)) : 0
                tg_sd = n > 1 ? sqrt((tg2 - tg * tg / n) / (n - 1)) : 0
                printf "%s\t%s\t%d\t%.2f\t%.2f\t%.2f\t%.2f\n", c, e, n, pp_mean, pp_sd, tg_mean, tg_sd
            }
        ' "$raw_file" >> "$summary_file"
    done
done

printf '\ncomplete: %s\n' "$summary_file"
column -t -s $'\t' "$summary_file" 2>/dev/null || awk -F '\t' '{ printf "%-8s %-7s %3s %14s %10s %13s %9s\n", $1, $2, $3, $4, $5, $6, $7 }' "$summary_file"

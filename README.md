# gemma4.c

Gemma 4 E2B inference in pure C.

## Quick start

Download and export the model:

```sh
python3 -m pip install -r requirements.txt
hf download google/gemma-4-E2B-it-qat-q4_0-unquantized --local-dir gemma4-qat
python3 exporter.py gemma4-qat -o gemma4-E2B-int8.bin
```

Build and run:

```sh
make
./run "Why is the sky blue?"
```

## Options

- The final argument sets the prompt.
- `-m PATH` sets the model path. The default is `gemma4-E2B-int8.bin`.
- `-t TEMPERATURE` sets the temperature. The default is `1.0`. Use `0` for greedy decoding.
- `-n TOKENS` sets the maximum number of tokens to generate. The default is `1,024`.
- `--dump-logits` writes the prompt logits to stdout as float32 binary data.

## Numerical validation

```sh
python3 validation.py
```

| Metric | Result |
| --- | ---: |
| Mean KL divergence | 0.004707 |
| Top-1 agreement | 96.2% (2297/2388) |
| Mean absolute error | 0.156529 |
| Mean cosine similarity | 0.998921 |

## Performance

```sh
make benchmark
./benchmark -m MODEL --pp PP --tg TG --d DEPTH
```

Profile without function inlining:

```sh
make profile
perf record -m 64 -e cycles:pp -- ./benchmark -m MODEL --pp PP --tg TG --d DEPTH
perf report
```

Benchmarked on an eight-core Ryzen 7 7700 using one OpenMP thread per physical core.
Each throughput result is the median of three runs after one discarded warmup.
perf cycle shares are from one run.

### Decode

| Starting context | Decode (tg128) | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 24.38&nbsp;tok&#8288;/&#8288;s | 65.98% | 26.21% | 1.40% | 1.29% |
| 1,024 | 24.07&nbsp;tok&#8288;/&#8288;s | 62.62% | 28.66% | 2.21% | 1.90% |
| 2,048 | 23.40&nbsp;tok&#8288;/&#8288;s | 58.86% | 30.34% | 3.29% | 3.23% |
| 4,096 | 22.06&nbsp;tok&#8288;/&#8288;s | 54.49% | 30.37% | 5.04% | 5.88% |

### Prefill

| Prompt length | Prefill | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 115.96&nbsp;tok&#8288;/&#8288;s | 57.44% | 35.86% | 1.39% | 1.28% |
| 1,024 | 113.75&nbsp;tok&#8288;/&#8288;s | 56.69% | 35.07% | 2.26% | 2.25% |
| 2,048 | 110.51&nbsp;tok&#8288;/&#8288;s | 55.29% | 34.21% | 3.28% | 3.49% |
| 4,096 | 104.65&nbsp;tok&#8288;/&#8288;s | 52.69% | 32.37% | 4.94% | 6.12% |

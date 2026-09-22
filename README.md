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
| Mean KL divergence | 0.004676 |
| Top-1 agreement | 96.4% (2302/2388) |
| Mean absolute error | 0.157333 |
| Mean cosine similarity | 0.998926 |

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
| 128 | 24.50&nbsp;tok&#8288;/&#8288;s | 77.73% | 12.35% | 2.55% | 0.46% |
| 512 | 23.11&nbsp;tok&#8288;/&#8288;s | 70.30% | 12.71% | 9.08% | 1.54% |
| 1,024 | 22.45&nbsp;tok&#8288;/&#8288;s | 64.47% | 12.40% | 14.68% | 2.27% |

### Prefill

| Prompt length | Prefill | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 89.05&nbsp;tok&#8288;/&#8288;s | 73.36% | 15.52% | 3.14% | 0.43% |
| 512 | 81.74&nbsp;tok&#8288;/&#8288;s | 67.15% | 14.26% | 11.25% | 1.64% |
| 1,024 | 75.01&nbsp;tok&#8288;/&#8288;s | 61.14% | 13.01% | 17.15% | 2.52% |

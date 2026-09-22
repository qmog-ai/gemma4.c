# gemma4.c

Gemma 4 E2B inference in pure C.

## Quick start

Export the model:

```sh
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-unquantized -o gemma4-E2B-int8.bin
```

Build and run:

```sh
make
./run "Why is the sky blue?"
```

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
`perf` cycle shares are from one run.

### Decode

| Starting context | Decode (tg128) | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 0 | 25.04 tok/s | 91.87% | 1.35% |
| 512 | 23.13 tok/s | 84.65% | 8.76% |
| 1,024 | 22.48 tok/s | 82.17% | 11.46% |

### Prefill

| Prompt length | Prefill | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 128 | 90.22 tok/s | 93.66% | 3.85% |
| 512 | 82.20 tok/s | 84.62% | 13.53% |
| 1,024 | 75.29 tok/s | 77.49% | 20.69% |

For a 512-token prompt, prefill becomes compute-bound around a batch size of four.

| Batch size | Prefill |
| ---: | ---: |
| 1 | 29.15 tok/s |
| 2 | 54.13 tok/s |
| 4 | 81.90 tok/s |
| 8 | 83.66 tok/s |
| 16 | 85.18 tok/s |
| 32 | 85.70 tok/s |
| 64 | 86.10 tok/s |
| 128 | 86.27 tok/s |
| 256 | 86.14 tok/s |
| 512 | 85.73 tok/s |

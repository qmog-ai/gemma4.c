# gemma4.c

Gemma 4 E2B inference in pure C.

## Quick start

```sh
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-unquantized -o gemma4-E2B-int8.bin
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
./benchmark -m MODEL --pp PP --tg TG --d DEPTH
```

Benchmarked on a Ryzen 7 7700 using eight OpenMP threads pinned to physical
cores. Each throughput result is the median of three runs after one discarded
warmup. `perf` cycle shares are from one run.

### Decode

| Starting context | Decode (tg128) | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 0 | 25.05 tok/s | 91.87% | 1.35% |
| 512 | 23.15 tok/s | 84.65% | 8.76% |
| 1,024 | 22.50 tok/s | 82.17% | 11.46% |

### Prefill

| Prompt length | Prefill | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 128 | 95.91 tok/s | 93.66% | 3.85% |
| 512 | 86.31 tok/s | 84.62% | 13.53% |
| 1,024 | 79.00 tok/s | 77.49% | 20.69% |

For a 512-token prompt, prefill becomes compute-bound around a batch size of four.

| Batch size | Prefill |
| ---: | ---: |
| 1 | 29.16 tok/s |
| 2 | 54.16 tok/s |
| 4 | 81.85 tok/s |
| 8 | 83.52 tok/s |
| 16 | 84.98 tok/s |
| 32 | 85.61 tok/s |
| 64 | 86.32 tok/s |
| 128 | 86.06 tok/s |
| 256 | 86.14 tok/s |
| 512 | 85.71 tok/s |

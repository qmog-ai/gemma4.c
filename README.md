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
python3 validation.py 512
```

| Metric | Result |
| --- | ---: |
| Mean KL divergence | 0.004557 |
| Top-1 agreement | 95.5% (489/512) |
| Mean absolute error | 0.155311 |
| Mean cosine similarity | 0.998817 |

## Performance

```sh
./benchmark -m MODEL --pp PP --tg TG --d DEPTH
```

Benchmarked on an eight-core Ryzen 7 7700 using one OpenMP thread per physical core.
Each result is the median of three runs after one discarded warmup.
`perf` cycle shares are from one run.

### Decode

| Starting context | Decode (tg128) | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 0 | 25.00 tok/s | 91.87% | 1.35% |
| 512 | 23.13 tok/s | 84.65% | 8.76% |
| 1,024 | 22.47 tok/s | 82.17% | 11.46% |

### Prefill

| Prompt length | Prefill | Matmul cycles | Attention cycles |
| ---: | ---: | ---: | ---: |
| 128 | 30.29 tok/s | 90.57% | 1.66% |
| 512 | 29.13 tok/s | 87.78% | 5.23% |
| 1,024 | 28.15 tok/s | 84.50% | 8.59% |

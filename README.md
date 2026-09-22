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
`perf` cycle shares are from one run with the three kernels temporarily marked `noinline`; this changed throughput by less than 0.3%.

### Decode

| Starting context | Decode (tg128) | `dot_i8` | `dot_f32` | Value sum |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 25.00 tok/s | 79.45% | 0.91% | 0.16% |
| 512 | 23.13 tok/s | 75.90% | 4.65% | 0.93% |
| 1,024 | 22.47 tok/s | 73.68% | 6.69% | 1.35% |

### Prefill

| Prompt length | Prefill | `dot_i8` | `dot_f32` | Value sum |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 30.29 tok/s | 78.65% | 1.15% | 0.19% |
| 512 | 29.13 tok/s | 76.65% | 3.99% | 0.75% |
| 1,024 | 28.15 tok/s | 73.99% | 6.45% | 1.30% |

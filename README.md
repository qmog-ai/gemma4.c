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
| Mean KL divergence | 0.004707 |
| Top-1 agreement | 96.2% (2297/2388) |
| Mean absolute error | 0.156529 |
| Mean cosine similarity | 0.998921 |

## Performance

```sh
./benchmark -m MODEL --pp PP --tg TG --d DEPTH
```

Benchmarked on an eight-core Ryzen 7 7700 using one OpenMP thread per physical core.
Each throughput result is the median of three runs after one discarded warmup.

### Decode

| Starting context | Decode (tg128) |
| ---: | ---: |
| 0 | 25.24 tok/s |
| 512 | 24.39 tok/s |
| 1,024 | 24.08 tok/s |
| 4,096 | 22.06 tok/s |

### Prefill

| Prompt length | Prefill |
| ---: | ---: |
| 128 | 118.40 tok/s |
| 512 | 115.32 tok/s |
| 1,024 | 113.12 tok/s |
| 4,096 | 104.32 tok/s |

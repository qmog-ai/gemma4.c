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
| 0 | 23.69 tok/s |
| 512 | 22.68 tok/s |
| 1,024 | 22.27 tok/s |
| 4,096 | 20.07 tok/s |

### Prefill

| Prompt length | Prefill |
| ---: | ---: |
| 128 | 118.83 tok/s |
| 512 | 116.46 tok/s |
| 1,024 | 114.18 tok/s |
| 4,096 | 107.19 tok/s |

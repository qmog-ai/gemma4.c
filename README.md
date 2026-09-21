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
python3 validation.py 64
```

| Metric | Result |
| --- | ---: |
| Mean KL divergence | 0.005276 |
| Top-1 agreement | 92.2% (59/64) |
| Mean absolute error | 0.199132 |
| Mean cosine similarity | 0.997996 |

## Performance

```sh
./benchmark -m MODEL --pp PP --tg TG --d DEPTH
```

Benchmarked on an eight-core Ryzen 7 7700 using one thread. Each result is the median of
three runs after one discarded warmup.

### Decode

| Starting context | Decode (tg128) |
| ---: | ---: |
| 0 | 10.16 tok/s |
| 256 | 9.03 tok/s |
| 512 | 8.25 tok/s |

### Prefill

| Prompt length | Prefill |
| ---: | ---: |
| 64 | 12.51 tok/s |
| 256 | 11.88 tok/s |
| 512 | 11.07 tok/s |

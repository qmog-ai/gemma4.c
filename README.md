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

| Length | Decode (tg 128) | Prefill |
| ---: | ---: | ---: |
| 512 | 24.36 tok/s | 117.01 tok/s |
| 1,024 | 24.01 tok/s | 114.17 tok/s |
| 2,048 | 23.32 tok/s | 110.75 tok/s |
| 4,096 | 22.03 tok/s | 104.70 tok/s |

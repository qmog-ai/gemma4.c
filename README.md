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
python3 validation.py 64
```

| Metric | Result |
| --- | ---: |
| Mean KL divergence | 0.002274 |
| Top-1 agreement | 95.3% (61/64) |
| Mean absolute error | 0.117840 |
| Mean cosine similarity | 0.999509 |

## Performance

```sh
make benchmark
./benchmark -m MODEL --pp 64,256,512 -r 3
./benchmark -m MODEL --tg 128 --d 64,256,512 -r 3
```

Profile one workload at a time without function inlining:

```sh
./profile.sh -m MODEL --tg 128 --d 512 -r 1
perf report --stdio --no-children --percent-limit 0
```

Benchmarked on an eight-core Ryzen 7 7700 using one thread. Throughput is the mean and sample
standard deviation of three runs after one discarded warmup. Profiles contain only the measured workload.

### Decode

| Starting context | Decode (tg128) | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 7.82 ± 0.03&nbsp;tok&#8288;/&#8288;s | 46.52% | 47.38% | 3.88% | 0.50% |
| 256 | 7.27 ± 0.00&nbsp;tok&#8288;/&#8288;s | 42.96% | 44.57% | 9.19% | 1.27% |
| 512 | 6.57 ± 0.00&nbsp;tok&#8288;/&#8288;s | 40.14% | 41.12% | 14.50% | 2.08% |

### Prefill

| Prompt length | Prefill | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 9.84 ± 0.04&nbsp;tok&#8288;/&#8288;s | 47.92% | 49.33% | 1.30% | 0.18% |
| 256 | 9.44 ± 0.02&nbsp;tok&#8288;/&#8288;s | 46.01% | 47.15% | 4.86% | 0.60% |
| 512 | 8.95 ± 0.04&nbsp;tok&#8288;/&#8288;s | 43.53% | 44.86% | 9.08% | 1.17% |

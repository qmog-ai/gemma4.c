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
| Mean KL divergence | 0.005276 |
| Top-1 agreement | 92.2% (59/64) |
| Mean absolute error | 0.199132 |
| Mean cosine similarity | 0.997996 |

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

Benchmarked on an eight-core Ryzen 7 7700 using one thread. Each result is the median of
three runs after one discarded warmup.

### Decode

| Starting context | Decode (tg128) | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 9.84&nbsp;tok&#8288;/&#8288;s | 76.87% | 16.16% | 4.20% | 0.53% |
| 256 | 9.03&nbsp;tok&#8288;/&#8288;s | 73.05% | 15.45% | 8.43% | 1.08% |
| 512 | 8.25&nbsp;tok&#8288;/&#8288;s | 68.71% | 14.47% | 13.26% | 1.67% |

### Prefill

| Prompt length | Prefill | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 12.51&nbsp;tok&#8288;/&#8288;s | 79.06% | 16.74% | 1.66% | 0.24% |
| 256 | 11.88&nbsp;tok&#8288;/&#8288;s | 75.46% | 15.70% | 6.27% | 0.79% |
| 512 | 11.07&nbsp;tok&#8288;/&#8288;s | 70.32% | 14.84% | 11.59% | 1.48% |

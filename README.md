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
Each result is the median of three runs after one discarded warmup.
perf cycle shares are from one run.

### Decode

| Starting context | Decode (tg128) | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.48&nbsp;tok&#8288;/&#8288;s | 78.99% | 11.87% | 1.84% | 0.33% |
| 512 | 23.09&nbsp;tok&#8288;/&#8288;s | 76.34% | 11.33% | 4.66% | 0.93% |
| 1,024 | 22.47&nbsp;tok&#8288;/&#8288;s | 74.11% | 10.91% | 6.75% | 1.38% |

### Prefill

| Prompt length | Prefill | dot_i8() | matmul_int8() | dot_f32() | weighted_sum() |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 30.23&nbsp;tok&#8288;/&#8288;s | 79.21% | 11.64% | 1.15% | 0.23% |
| 512 | 29.12&nbsp;tok&#8288;/&#8288;s | 76.89% | 11.37% | 4.02% | 0.76% |
| 1,024 | 28.13&nbsp;tok&#8288;/&#8288;s | 74.35% | 10.92% | 6.47% | 1.33% |

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
- `--thinking` enables the model's thinking chat template.

## Correctness

The C tokenizer matches the official Hugging Face tokenizer across raw text, chat templates, thinking templates, Unicode, byte fallback, and special tokens.

### Numerical validation

All 2,388 positions are compared against the unquantized QAT checkpoint over the complete 262,144-token vocabulary.

| Metric | Result |
| --- | ---: |
| Mean absolute error | 0.127127 |
| Maximum absolute error | 8.203652 |
| Mean cosine similarity | 0.999307 |
| Top-1 agreement | 97.0% (2,317/2,388) |
| Mean KL divergence | 0.003484 |

### Evaluation

| Runtime | tinyMMLU |
| --- | ---: |
| Hugging Face Gemma 4 E2B IT | 60/100 |
| Hugging Face unquantized QAT checkpoint | 56/100 |
| gemma4.c | 56/100 |

Commands for reproducing the tokenizer, numerical validation, and evaluation results are in [tests](tests/README.md).

## Performance

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

Benchmark and profiling commands are in [performance](performance/README.md).

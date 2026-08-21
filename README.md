# gemma4.c

Gemma 4 inference in a single C file.

`exporter.py` converts a Hugging Face checkpoint into the binary model format loaded by the runtime.

## Benchmark

Measured on a Ryzen 7 7700 using eight threads. The benchmark uses 32 fixed
prompt tokens followed by 16 decode steps. Results are the median of three
runs after one warmup, excluding model loading and tokenization.

Correctness was compared with the finished packed runtime over 512 tokens and
the full vocabulary using `KL(packed || OpenMP)`.

| Mean KL | Max KL | Top-1 agreement | Prefill | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 0.002841 | 0.034903 | 97.66% | 5.45 tok/s | 4.52 tok/s |

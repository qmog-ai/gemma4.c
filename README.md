# gemma4.c

Gemma 4 inference in a single C file.

`exporter.py` converts a Hugging Face checkpoint into the binary model format loaded by the runtime.

## Benchmark

Measured on a Ryzen 7 7700 using one thread. The benchmark uses 32 fixed
prompt tokens followed by 16 decode steps. Results are the median of three
runs after one warmup, excluding model loading and tokenization.

Correctness was compared with the finished packed runtime over 64 tokens and
the full vocabulary using `KL(packed || scalar)`.

| Mean KL | Max KL | Top-1 agreement | Prefill | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 0.002842 | 0.018807 | 100% | 0.75 tok/s | 0.63 tok/s |

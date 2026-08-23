# gemma4.c

Gemma 4 inference in a single C file.

`exporter.py` converts a Hugging Face checkpoint into the binary model format loaded by the runtime.

## Benchmark

Measured on a Ryzen 7 7700 using eight threads. The benchmark uses 512 fixed
prompt tokens followed by 16 decode steps. Results are the median of three
runs after one warmup, excluding model loading and tokenization.

Correctness was compared with the unquantized Hugging Face checkpoint over
2,388 tokens and the full vocabulary using `KL(HF || runtime)`.

| Mean KL | Max KL | Top-1 agreement | Prefill | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 0.005207 | 1.199355 | 96.5% | 625.61 tok/s | 25.22 tok/s |

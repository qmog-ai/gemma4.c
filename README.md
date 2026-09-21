# gemma4.c

Gemma 4 inference in a single C file.

`exporter.py` converts a Hugging Face checkpoint into the binary model format loaded by the runtime. Matrix weights use int8 values with one FP32 scale per group of 64 values. Linear inputs are quantized to the same format at runtime.

## Benchmark

Measured on a Ryzen 7 7700 using one thread. The benchmark uses 32 fixed
prompt tokens followed by 16 decode steps. Results are the median of three
runs after one warmup, excluding model loading and tokenization.

Correctness was compared with the former finished FP16-scale runtime over 64
tokens and the full vocabulary using `KL(packed FP16 || scalar FP32)`.

| Mean KL | Max KL | Top-1 agreement | Prefill | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 0.006024 | 0.095781 | 95.31% | 12.41 tok/s | 10.03 tok/s |

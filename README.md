# gemma4.c

Gemma 4 inference in a single C file.

`exporter.py` converts a Hugging Face checkpoint into the binary model format loaded by the runtime. Matrix weights use int8 values with one FP32 scale per group of 64 values.

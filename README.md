# gemma4.c

`exporter.py` converts a Hugging Face Gemma 4 E2B checkpoint into a compact
binary model file. Two-dimensional weight tensors are quantized in groups of
64 signed 8-bit values, with one FP32 scale per group.

`gemma4.c` memory-maps the model file and resolves its stored offsets into
pointers.

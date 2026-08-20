# gemma4.c

An experimental Gemma 4 E2B inference implementation in C.

The first step is `exporter.py`, which converts the Hugging Face checkpoint into a compact binary layout that the C runtime can read directly.

```bash
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized -o ./gemma4-E2B-int8.bin
```

# gemma4-e2b.c

Gemma 4 E2B inference in 770 lines of C.

A minimal educational implementation of Gemma 4 inference on CPU, inspired by Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c). The goal is readability and understanding, not performance or production use.

<img width="800" height="299" alt="demo" src="https://github.com/user-attachments/assets/a39e9809-050a-4f47-a978-83d04c312146" />

## Benchmark

Measured on an AMD Ryzen 7 7700 (8 cores).

| Implementation | Prefill (pp512) | Decode (tg128) |
| -------------- | --------------: | -------------: |
| gemma4-e2b.c (int8) | 638.86 ± 2.93 tok/s | 25.90 ± 0.01 tok/s |
| llama.cpp (Q8_0) | 276.32 ± 1.85 tok/s | 23.83 ± 0.01 tok/s |

Results are the mean ± sample standard deviation over 12 timed runs after one discarded warmup, using the default native build.
```bash
python3 benchmark.py [prefill] [decode] [runs]
```
## Numerical validation

The implementation is validated against Google's unquantized [Gemma 4 E2B QAT checkpoint](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) using Hugging Face Transformers. Both implementations process the same 1,024-token passage under teacher forcing, and their logits are compared at every position.

| Metric | Result |
| ------ | -----: |
| Top-1 agreement | 1,002 / 1,024 (97.9%) |
| Mean KL divergence | 0.003489 |

An exact numerical match is not expected here because `gemma4-e2b.c` quantizes the model to int8, while the reference runs the unquantized checkpoint in BF16. Despite the quantization, the output distributions remain closely aligned.

```bash
python3 validation.py [n_tokens]
```

## Model

### Option 1: Download the exported model

The easiest option is to download the ready-to-run model file. It is about 5.2 GiB.

```bash
python3 -m pip install -U huggingface_hub
hf download username/gemma4-e2b-int8 gemma4-E2B-int8.mog --local-dir .
```

### Option 2: Export from a checkpoint

This runtime reads its own single-file `.mog` format. `exporter.py` converts the original Hugging Face checkpoint into a file the C program can load. As part of that conversion, it quantizes the model's weights to int8 to make the model smaller and faster on CPU.

```bash
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized -o ./gemma4-E2B-int8.mog
```

Python is only needed to export the model or run numerical validation. Once the `.mog` file exists, inference runs entirely through the C program.

## Usage

Running the model requires Linux on an x86-64 CPU with AVX2, an OpenMP-capable C compiler, and `make`. 8 GiB of RAM is recommended.

```bash
make

# Generate text
./run -m ./gemma4-E2B-int8.mog -t 1.0 -n 256 "Why is the sky blue?"

# Benchmark prefill and decode
./run -m ./gemma4-E2B-int8.mog --bench 512 128

# Dump prompt logits
./run -m ./gemma4-E2B-int8.mog -n 64 --dump-logits "Why is the sky blue?" > logits.bin
```

- `-m`: model path; defaults to `gemma4-E2B-int8.mog`.
- `-t`: temperature; defaults to `1.0`, or `0` for greedy decoding.
- `-n`: maximum sequence length; defaults to `1,024` tokens.
- `--bench`: benchmark prefill and decode throughput.
- `--dump-logits`: write prompt logits as float32 binary data.

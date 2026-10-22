# Tests

Build the shared library used by the Python tests:

```sh
make libgemma4.so
```

Compare the C tokenizer with the official Hugging Face tokenizer:

```sh
python3 tests/tokenizer.py --model MODEL --tokenizer CHECKPOINT
```

Compare logits against the Hugging Face model after every known input token. Pass a token count for a shorter run or omit it for all 2,388 tokens:

```sh
python3 tests/validation.py 64 --model MODEL --checkpoint CHECKPOINT
python3 tests/validation.py --model MODEL --checkpoint CHECKPOINT
```

Run the frozen 100-question tinyMMLU evaluation:

```sh
python3 tests/eval.py --model MODEL --tokenizer CHECKPOINT
```

Thinking mode greedily generates a reasoning trace before scoring the answer. Use `--limit` for a shorter run:

```sh
python3 tests/eval.py --thinking --limit 1 --model MODEL --tokenizer CHECKPOINT
```

Evaluation results are written under `tests/results/`.

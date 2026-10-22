#!/usr/bin/env python3
"""Compare the gemma4.c tokenizer with the official Hugging Face tokenizer."""

import argparse
import ctypes
from pathlib import Path

from transformers import AutoTokenizer

from bindings import load_library


ROOT = Path(__file__).resolve().parent.parent
CAPACITY = 4096
COMPLEX_PROMPT = (
    "Montréal café — “naïve” 中文 العربية 👩🏽‍💻\n"
    "Tabs\tand  spaces; <angle> <|image|> bytes: 𐍈."
)
THINKING_PROMPT = "A night shift has 12 aisles to stock."


def tokenize_c(lib, model, mode: str, prompt: str) -> list[int]:
    output = (ctypes.c_int32 * CAPACITY)()
    count = lib.tokenize_prompt(
        model,
        prompt.encode(),
        mode == "thinking",
        mode == "raw",
        output,
        CAPACITY,
    )
    if count < 0:
        raise AssertionError(f"tokenizer capacity exceeded for {prompt!r}")
    return list(output[:count])


def tokenize_hf(tokenizer, mode: str, prompt: str) -> list[int]:
    if mode == "raw":
        return [tokenizer.bos_token_id] + tokenizer.encode(
            prompt, add_special_tokens=False
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=mode == "thinking",
    )["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "libgemma4.so")
    args = parser.parse_args()

    lib = load_library(args.library)
    model_size = ctypes.c_size_t()
    model = lib.load_model(str(args.model).encode(), ctypes.byref(model_size))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tests = {
        "raw": [COMPLEX_PROMPT, "", *tokenizer.get_added_vocab()],
        "chat": ["Hello, world!"],
        "thinking": [THINKING_PROMPT],
    }

    try:
        for mode, prompts in tests.items():
            for prompt in prompts:
                expected = tokenize_hf(tokenizer, mode, prompt)
                actual = tokenize_c(lib, model, mode, prompt)
                if actual != expected:
                    raise AssertionError(
                        f"{mode} {prompt!r}:\nexpected {expected}\nactual   {actual}"
                    )

        output = (ctypes.c_int32 * 1)()
        count = lib.tokenize_prompt(
            model,
            b"x",
            False,
            True,
            output,
            1,
        )
        if count != -1:
            raise AssertionError("capacity: expected failure")
    finally:
        lib.free_model(model, model_size)

    print("tokenizer ok")


if __name__ == "__main__":
    main()

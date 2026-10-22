#!/usr/bin/env python3
"""Score gemma4.c on the frozen 100-question tinyMMLU evaluation."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from bindings import load_library


ROOT = Path(__file__).resolve().parent.parent
LABELS = "ABCD"
THOUGHT_END = 101  # <channel|>


def forward(lib, model, state, tokens: list[int], position: int) -> int:
    for token in tokens:
        token = ctypes.c_int32(token)
        lib.forward(model, state, ctypes.byref(token), 1, position)
        position += 1
    return position


def answer_logits(
    lib, model, state, tokens: list[int], thinking: bool, max_context: int, vocab_size: int
) -> np.ndarray:
    position = forward(lib, model, state, tokens, 0)
    if not thinking:
        scores = lib.logits(model, state, 0)
        return np.ctypeslib.as_array(scores, shape=(vocab_size,)).copy()

    for _ in range(max_context - len(tokens)):
        scores = lib.logits(model, state, 0)
        logits = np.ctypeslib.as_array(scores, shape=(vocab_size,))
        token = int(np.argmax(logits))
        position = forward(lib, model, state, [token], position)
        if token == THOUGHT_END:
            scores = lib.logits(model, state, 0)
            return np.ctypeslib.as_array(scores, shape=(vocab_size,)).copy()
    raise RuntimeError("thinking did not reach the final answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "libgemma4.so")
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/tiny_mmlu.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    questions = json.loads(args.dataset.read_text())
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        questions = questions[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefix = "" if args.thinking else " "
    choices = [tokenizer.encode(prefix + label, add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in choices):
        raise SystemExit(f"answer labels are not single tokens: {choices}")
    choice_ids = [tokens[0] for tokens in choices]

    lib = load_library(args.library)
    model_size = ctypes.c_size_t()
    model = lib.load_model(str(args.model).encode(), ctypes.byref(model_size))
    state = lib.create_state()
    if not state:
        lib.free_model(model, model_size)
        raise MemoryError("could not allocate inference state")

    results = []
    correct = 0
    try:
        for index, question in enumerate(questions, 1):
            if args.thinking:
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question["prompt"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )["input_ids"]
            else:
                tokens = [tokenizer.bos_token_id]
                tokens += tokenizer.encode(question["prompt"], add_special_tokens=False)

            logits = answer_logits(
                lib,
                model,
                state,
                tokens,
                args.thinking,
                tokenizer.model_max_length,
                len(tokenizer),
            )
            scores = [float(logits[token]) for token in choice_ids]
            prediction = int(np.argmax(scores))
            answer = int(question["answer"])
            correct += prediction == answer
            results.append(
                {
                    "id": question["id"],
                    "answer": LABELS[answer],
                    "prediction": LABELS[prediction],
                    "logits": dict(zip(LABELS, scores)),
                }
            )
            print(
                f"[{index:3d}/{len(questions)}] "
                f"{LABELS[prediction]}/{LABELS[answer]} "
                f"{'correct' if prediction == answer else 'wrong'}",
                flush=True,
            )
    finally:
        lib.free_state(state)
        lib.free_model(model, model_size)

    output = args.output or ROOT / "tests/results" / (
        "thinking.json" if args.thinking else "eval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"correct": correct, "results": results}, indent=2) + "\n")
    print(f"accuracy {correct / len(questions) * 100:.1f}% ({correct}/{len(questions)})")
    print(f"results {output}")


if __name__ == "__main__":
    main()

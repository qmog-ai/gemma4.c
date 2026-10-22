#!/usr/bin/env python3
"""Compare gemma4.c logits with the Hugging Face BF16 reference."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
from pathlib import Path

import numpy as np

from bindings import load_library


ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
DEFAULT_TOKENS = 2388


def load_reference(checkpoint: str, tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text = (ROOT / "tests/validation.txt").read_text()
    ids = [tokenizer.bos_token_id]
    ids += tokenizer.encode(text, add_special_tokens=False)[: tokens - 1]
    if len(ids) != tokens:
        raise SystemExit(f"validation text contains only {len(ids)} tokens")

    options = {"dtype": "auto"}
    if importlib.util.find_spec("accelerate") is not None:
        options["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **options)
    except Exception:
        model = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, **options)
    model.eval()

    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        expected = model(input_ids=input_ids).logits[0].float().cpu().numpy()
    return np.asarray(ids, dtype=np.int32), expected


def compare(lib, model, state, ids: np.ndarray, expected: np.ndarray) -> None:
    absolute_sum = 0.0
    maximum_error = 0.0
    cosine_sum = 0.0
    top1 = 0
    kl_sum = 0.0

    for position, (token, reference) in enumerate(zip(ids, expected)):
        token = ctypes.c_int32(token)
        lib.forward(model, state, ctypes.byref(token), 1, position)
        scores = lib.logits(model, state, 0)
        actual32 = np.ctypeslib.as_array(scores, shape=(expected.shape[1],))
        top1 += int(actual32.argmax() == reference.argmax())
        actual = actual32.astype(np.float64)
        reference = reference.astype(np.float64)

        error = np.abs(actual - reference)
        absolute_sum += float(error.sum())
        maximum_error = max(maximum_error, float(error.max()))
        cosine_sum += float(
            np.dot(actual, reference)
            / max(np.linalg.norm(actual) * np.linalg.norm(reference), 1e-12)
        )

        actual -= actual.max()
        reference -= reference.max()
        actual_probability = np.exp(actual)
        reference_probability = np.exp(reference)
        actual_probability /= actual_probability.sum()
        reference_probability /= reference_probability.sum()
        kl_sum += float(
            np.sum(
                reference_probability
                * np.log((reference_probability + 1e-12) / (actual_probability + 1e-12))
            )
        )

    count, vocab = expected.shape
    print(f"mean absolute error     {absolute_sum / (count * vocab):.6f}")
    print(f"maximum absolute error  {maximum_error:.6f}")
    print(f"mean cosine similarity  {cosine_sum / count:.6f}")
    print(f"top-1 agreement         {top1 / count * 100:.1f}% ({top1}/{count})")
    print(f"mean KL divergence      {kl_sum / count:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="?", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", default=HF_REPO)
    parser.add_argument("--library", type=Path, default=ROOT / "libgemma4.so")
    args = parser.parse_args()

    if not 1 <= args.tokens <= DEFAULT_TOKENS:
        raise SystemExit(f"tokens must be between 1 and {DEFAULT_TOKENS}")
    ids, expected = load_reference(args.checkpoint, args.tokens)

    lib = load_library(args.library)
    model_size = ctypes.c_size_t()
    model = lib.load_model(str(args.model).encode(), ctypes.byref(model_size))
    state = lib.create_state()
    if not state:
        lib.free_model(model, model_size)
        raise MemoryError("could not allocate inference state")
    try:
        compare(lib, model, state, ids, expected)
    finally:
        lib.free_state(state)
        lib.free_model(model, model_size)


if __name__ == "__main__":
    main()

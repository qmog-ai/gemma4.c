#!/usr/bin/env python3
"""Numerically validate gemma4.c against the Hugging Face Transformers reference implementation.

What this script does
---------------------
The reference is Google's official unquantized Gemma 4 E2B QAT checkpoint,
executed in its native BF16 dtype by Hugging Face Transformers. The implementation under
test is gemma4.c using the exported int8 .bin model.

The script runs the opening of the Éva Gauthier article in the
WikiText-103 raw validation split through gemma4.c and Hugging Face
Transformers, then compares their logits at every position. Top-1 agreement
and KL divergence show how closely the int8 C implementation tracks the
reference. Absolute error and cosine similarity provide more detail when
investigating differences.

Setup
-----
1. Put gemma4-E2B-int8.bin in the repository root. To create it, run:

     python3 exporter.py ../models/gemma-4-E2B-it-qat-q4_0-unquantized \\
       -o gemma4-E2B-int8.bin

2. Install Python packages needed for a fresh Hugging Face reference:

     python3 -m pip install numpy torch transformers safetensors

3. Place the unquantized checkpoint in the repository root or at
   ../models/gemma-4-E2B-it-qat-q4_0-unquantized. Otherwise, Transformers
   downloads google/gemma-4-E2B-it-qat-q4_0-unquantized from Hugging Face.
   The 2,388-token BF16 run peaks at roughly 10 GiB of RAM. The C executable is
   built automatically with make when it is missing or out of date.

Reproduce the README result
---------------------------
Run the validation. It compares the complete article by default: 2,387 text
tokens plus BOS, for 2,388 model positions.

  python3 validation.py

Pass a smaller token count for a quicker comparison:

  python3 validation.py 64
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

HF_REPO = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
ROOT = Path(__file__).resolve().parent
LOCAL_MODELS = (
    ROOT / "gemma-4-E2B-it-qat-q4_0-unquantized",
    ROOT.parent / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized",
)
DEFAULT_TOKENS = 2388
MAX_TOKENS = 2388

VALIDATION_TEXT = ROOT / "validation.txt"


def default_model():
    for path in LOCAL_MODELS:
        if path.is_dir():
            return str(path)
    return HF_REPO


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, Gemma4ForConditionalGeneration

    kwargs = dict(dtype="auto")
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def tokenize_prompt(tok, text: str, tokens: int):
    content = tok.encode(text, add_special_tokens=False)
    if len(content) + 1 < tokens:
        raise SystemExit(f"text is only {len(content) + 1} tokens, need {tokens}")
    ids = [tok.bos_token_id] + content[: tokens - 1]
    prompt = tok.decode(ids[1:])
    again = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
    if again != ids:
        raise SystemExit("truncated prompt does not round-trip through the tokenizer")
    return prompt, np.array(ids, dtype=np.int32)


def generate_hf_logits(model_id: str, text: str, tokens: int):
    import torch

    tok = load_tokenizer(model_id)
    prompt, ids = tokenize_prompt(tok, text, tokens)
    print(f"loading {model_id}", file=sys.stderr)
    model = load_model(model_id)
    print(f"forward in {next(model.parameters()).dtype}, no weight quantization", file=sys.stderr)
    device = next(model.parameters()).device
    x = torch.tensor([ids.tolist()], dtype=torch.long, device=device)
    print(f"forward {len(ids)} tokens", file=sys.stderr)
    with torch.inference_mode():
        logits = model(input_ids=x).logits[0].float().cpu().numpy().astype(np.float32)
    return prompt, ids, logits


def ensure_run(run: Path):
    src = ROOT / "gemma4.c"
    if run.is_file() and run.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(["make", "-C", str(ROOT)])


def dump_runtime_logits(run: Path, model: Path, prompt: str) -> np.ndarray:
    proc = subprocess.run(
        [str(run), "-m", str(model), "--dump-logits", prompt],
        stdout=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def report(actual: np.ndarray, expected: np.ndarray):
    n_tokens, vocab = expected.shape
    if actual.size != n_tokens * vocab:
        raise SystemExit(
            f"shape mismatch: expected {n_tokens}x{vocab} floats, got {actual.size}"
        )
    actual = actual.reshape(n_tokens, vocab)

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    # A full comparison contains well over two GiB of float32 logits per side.
    # Stream rows so float64 metric temporaries do not multiply that footprint.
    abs_sum = 0.0
    max_abs = 0.0
    cosine_sum = 0.0
    top1_count = 0
    kl_sum = 0.0
    rows_per_chunk = 8
    for start in range(0, n_tokens, rows_per_chunk):
        stop = min(n_tokens, start + rows_per_chunk)
        a = actual[start:stop].astype(np.float64)
        e = expected[start:stop].astype(np.float64)

        abs_err = np.abs(a - e)
        abs_sum += float(abs_err.sum())
        max_abs = max(max_abs, float(abs_err.max()))

        num = (a * e).sum(axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(e, axis=-1)
        cosine_sum += float((num / np.maximum(den, 1e-12)).sum())
        top1_count += int(
            (actual[start:stop].argmax(axis=-1) == expected[start:stop].argmax(axis=-1)).sum()
        )

        p_g = softmax(e)
        p_c = softmax(a)
        kl_sum += float(
            (p_g * (np.log(p_g + 1e-12) - np.log(p_c + 1e-12))).sum()
        )

    mae = abs_sum / (n_tokens * vocab)
    cosine = cosine_sum / n_tokens
    top1 = top1_count / n_tokens
    mean_kl = kl_sum / n_tokens

    print(f"mean absolute error     {mae:.6f}")
    print(f"maximum absolute error  {max_abs:.6f}")
    print(f"mean cosine similarity  {cosine:.6f}")
    print(f"top-1 agreement         {top1 * 100:.1f}% ({top1_count}/{n_tokens})")
    print(f"mean KL divergence      {mean_kl:.6f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("tokens", nargs="?", type=int, default=DEFAULT_TOKENS)
    p.add_argument("--checkpoint", default=default_model())
    p.add_argument("--model", type=Path, default=ROOT / "gemma4-E2B-int8.bin")
    p.add_argument("--run", type=Path, default=ROOT / "run")
    args = p.parse_args()

    if args.tokens < 1 or args.tokens > MAX_TOKENS:
        raise SystemExit(f"tokens must be between 1 and {MAX_TOKENS}")
    if not args.model.is_file():
        raise SystemExit(f"missing model file: {args.model}")

    prompt, _ids, expected = generate_hf_logits(
        args.checkpoint,
        VALIDATION_TEXT.read_text(encoding="utf-8"),
        args.tokens,
    )
    ensure_run(args.run)
    actual = dump_runtime_logits(args.run, args.model, prompt)
    report(actual, expected.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()

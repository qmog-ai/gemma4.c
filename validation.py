#!/usr/bin/env python3
"""Numerically validate gemma4-e2b.c against the model it was quantized from.

What this script does
---------------------
The reference is Google's official unquantized Gemma 4 E2B QAT checkpoint,
executed in float32 by Hugging Face Transformers. The implementation under
test is gemma4-e2b.c using the exported int8 .mog model.

Both models receive exactly the same fixed passage. This is called teacher
forcing: each position sees the correct preceding text instead of tokens that
either model generated. A disagreement at one position therefore cannot alter
the input at later positions. gemma4-e2b.c dumps its complete 262,144-token logit
vector at every position, and this script compares those vectors with the
Hugging Face reference. It reports absolute error, cosine similarity, top-1
agreement, and KL divergence.

Setup
-----
1. Put gemma4-E2B-int8.mog in the repository root. To create it, run:

     python3 exporter.py ../models/gemma-4-E2B-it-qat-q4_0-unquantized \\
       -o gemma4-E2B-int8.mog

2. Install Python packages needed for a fresh Hugging Face reference:

     python3 -m pip install numpy torch transformers safetensors

3. Place the unquantized checkpoint in the repository root or at
   ../models/gemma-4-E2B-it-qat-q4_0-unquantized. Otherwise, Transformers
   downloads google/gemma-4-E2B-it-qat-q4_0-unquantized from Hugging Face.
   The 1,024-token float32 run needs roughly 30 GB of RAM. The C executable is
   built automatically with make when it is missing or out of date.

Reproduce the README result
---------------------------
Run the validation. It compares 1,024 tokens by default:

  python3 validation.py

Pass a smaller token count for a quicker comparison:

  python3 validation.py 64

Passing --dump saves the Hugging Face reference for reuse. A 1,024-token dump
is about 1 GB and should not be committed to the repository.
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
DEFAULT_TOKENS = 1024
MAX_TOKENS = 1024

# Long enough that BOS + BPE is at least 1024 tokens; truncated to --tokens.
DEFAULT_TEXT = (
    "The sky appears blue during the day because sunlight is scattered by the "
    "molecules of the atmosphere, and shorter wavelengths scatter far more "
    "strongly than longer ones. This effect is called Rayleigh scattering, and "
    "it grows with the inverse fourth power of the wavelength, so blue light "
    "at 450 nanometers is scattered nearly ten times more than red light at "
    "700 nanometers. At sunset the light must travel through a much longer "
    "column of air, so most of the blue is scattered away before it reaches "
    "the observer, leaving the reds and oranges that paint the horizon. The "
    "capital of France is Paris, a city on the Seine founded more than two "
    "thousand years ago on an island called the Ile de la Cite. Paris grew "
    "from a Roman town named Lutetia into the political and cultural center "
    "of the French state, and today it holds about two million people within "
    "its ring road and more than ten million in its wider urban area. The "
    "city is known for the Eiffel Tower, completed in 1889 for the World "
    "Fair, for the Louvre, which began as a medieval fortress and is now the "
    "most visited museum in the world, and for Notre-Dame cathedral, whose "
    "restoration after the 2019 fire took more than five years of careful "
    "work by carpenters, masons, and glaziers using both medieval and modern "
    "techniques. French cuisine, fashion, and cinema all radiate outward from "
    "the capital, and the language spoken there sets the standard for French "
    "around the world. "
    "Gemma 4 is a dense decoder-only transformer. This educational runtime "
    "keeps the whole forward pass in one C file: embedding, per-layer inputs, "
    "attention with rotary positions, a gated MLP, and a tied embedding used "
    "as the classifier. Weights are stored as grouped int8 with a scale per "
    "block of sixty four elements, and the packed layout writes sixteen output "
    "rows together so a cache line of weights is reused across nearby outputs. "
    "On hosts with AVX-512 VNNI the inner product uses vpdpbusd; otherwise the "
    "same arithmetic runs in AVX2. OpenMP parallelizes over output tiles and "
    "defaults to one worker per physical core unless OMP_NUM_THREADS is set. "
    "Prefill walks the prompt in chunks of five hundred twelve tokens and "
    "writes keys and values into a cache that later decode steps read. Shared "
    "layers reuse an earlier cache owner so the memory footprint stays closer "
    "to the number of distinct attention patterns than to the layer count. "
    "A comparison should not demand bit-exact equality with Hugging Face. "
    "The reference checkpoint is the unquantized QAT model run in float32; "
    "gemma4-e2b.c requantizes activations every matmul and uses a lookup table "
    "for GeGLU. Cosine similarity near one and unchanged top-1 tokens are "
    "the practical bar. Mean absolute error on the order of a tenth of a "
    "logit is the drift we record in the README. "
    "Bangkok sits on the Chao Phraya river plain and has been the capital of "
    "Thailand since the late eighteenth century. The city is a port, a rail "
    "hub, and a center for rice, tourism, and electronics assembly. Street "
    "food, Buddhist temples, and a dense canal network sit beside elevated "
    "rail lines and a growing subway. Further north the mountains give way "
    "to the central plains that feed much of the country, and to the south "
    "the peninsula meets the Andaman Sea and the Gulf of Thailand. "
    "A reproducible dump always starts from the same English passage, cuts it "
    "to exactly N tokens after the BOS id, and decodes that prefix back to "
    "text so any compatible tokenizer yields the same ids. gemma4-e2b.c then "
    "encodes that text with its own BPE tables from the mog file. If those "
    "two sequences diverge, the comparison is invalid and this script exits. "
    "The validation uses one thousand twenty four tokens, crossing the "
    "five hundred twelve token prefill boundary while remaining small enough "
    "for a practical reference run. Generating Hugging Face logits loads the "
    "full unquantized checkpoint and creates a dense logit tensor, so the "
    "comparison is deliberately memory heavy. "
    "Integer matrix multiplication on this chip issues VNNI dots over "
    "unsigned bytes, so activations are biased by one hundred twenty eight "
    "and a correction term restores the signed product. The exporter packs "
    "sixteen output rows so each sixty four byte weight vector is reused "
    "across a tile. After the last layer the residual is RMS-normalized, "
    "projected through the tied embedding, and soft-capped with a tanh. "
    "Those classifier scores are the logits this script compares. Decode "
    "speed is limited by reading about two point three gigabytes of active "
    "weights per token; prefill is limited by arithmetic and scales with "
    "AVX-512. The benchmark script next to this file measures that split "
    "with twelve timed runs after one warmup. "
    "When you add a kernel, keep this comparison in the loop. A silent "
    "change in top-1 on this passage deserves investigation. A small move in "
    "mean absolute error is usually just another rounding path. "
    "Copper is a reddish metal that conducts heat and electricity well. It "
    "was one of the first metals people worked, and with tin it makes bronze. "
    "Today most copper is used in wiring, motors, and plumbing. Chile, Peru, "
    "and the United States mine a large share of the world's supply. Recycling "
    "scrap copper uses far less energy than smelting new ore. "
    "A year has about three hundred sixty five days. Earth travels around the "
    "Sun once in that time, and the tilt of its axis gives the seasons. Leap "
    "years add a day in February so the calendar does not drift. The Moon "
    "takes a little over twenty seven days to orbit Earth, and the cycle of "
    "phases that we see is about twenty nine and a half days. Tides rise and "
    "fall mainly because of the Moon's gravity, with a smaller pull from the "
    "Sun. Coastal cities plan harbors and flood walls around those rhythms. "
    "Libraries keep books, maps, and now digital records. The idea is simple: "
    "write things down so the next person does not have to discover them "
    "again. This repository does the same for one small inference stack."
)


def default_model():
    for path in LOCAL_MODELS:
        if path.is_dir():
            return str(path)
    return HF_REPO


def default_out(tokens: int) -> Path:
    return ROOT / f"gemma4_golden_{tokens}.npz"


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
    bos = tok.bos_token_id
    ids = [bos] + tok.encode(text, add_special_tokens=False)
    if len(ids) < tokens:
        raise SystemExit(f"text is only {len(ids)} tokens, need {tokens}")
    ids = ids[:tokens]
    prompt = tok.decode(ids[1:])
    again = [bos] + tok.encode(prompt, add_special_tokens=False)
    if again != ids:
        raise SystemExit("truncated prompt does not round-trip through the tokenizer")
    return prompt, np.array(ids, dtype=np.int32)


def generate_hf_logits(model_id: str, text: str, tokens: int):
    import torch

    tok = load_tokenizer(model_id)
    prompt, ids = tokenize_prompt(tok, text, tokens)
    print(f"loading {model_id}", file=sys.stderr)
    model = load_model(model_id).float()
    print("forward in float32, no weight quantization", file=sys.stderr)
    device = next(model.parameters()).device
    x = torch.tensor([ids.tolist()], dtype=torch.long, device=device)
    print(f"forward {len(ids)} tokens", file=sys.stderr)
    with torch.inference_mode():
        logits = model(input_ids=x).logits[0].float().cpu().numpy().astype(np.float32)
    return prompt, ids, logits


def load_or_build_golden(path: Path, tokens: int, model_id: str, text: str, dump: bool):
    if path.is_file():
        g = np.load(path)
        n = int(g["logits"].shape[0])
        if n >= tokens and "ids" in g.files:
            ids = np.asarray(g["ids"], dtype=np.int32)[:tokens]
            logits = g["logits"].astype(np.float32, copy=False)[:tokens]
            if n == tokens:
                prompt = str(g["prompt"])
                print(f"using {path} ({tokens} tokens)", file=sys.stderr)
                return prompt, ids, logits
            tok = load_tokenizer(model_id)
            prompt, sliced = tokenize_prompt(tok, tok.decode(ids[1:]), tokens)
            if not np.array_equal(sliced, ids):
                raise SystemExit("sliced golden ids do not round-trip")
            print(f"using first {tokens} tokens of {path}", file=sys.stderr)
            return prompt, ids, logits

    prompt, ids, logits = generate_hf_logits(model_id, text, tokens)
    if dump:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, prompt=prompt, ids=ids, logits=logits)
        print(f"wrote {path}: {logits.shape[0]} tokens x {logits.shape[1]} vocab", file=sys.stderr)
    return prompt, ids, logits


def ensure_run(run: Path):
    src = ROOT / "gemma4-e2b.c"
    if run.is_file() and run.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(["make", "-C", str(ROOT)])


def dump_runtime_logits(run: Path, mog: Path, prompt: str) -> np.ndarray:
    proc = subprocess.run(
        [str(run), "-m", str(mog), "-n", "8192", "--dump-logits", prompt],
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

    # A 1024-token comparison contains one GiB of float32 logits per side.
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
    p.add_argument("--dump", action="store_true", help="write the HF reference to --out")
    p.add_argument("--out", type=Path, default=None, help="golden npz path")
    p.add_argument("--model", default=default_model())
    p.add_argument("--mog", type=Path, default=ROOT / "gemma4-E2B-int8.mog")
    p.add_argument("--run", type=Path, default=ROOT / "run")
    args = p.parse_args()

    if args.tokens < 1 or args.tokens > MAX_TOKENS:
        raise SystemExit(f"tokens must be between 1 and {MAX_TOKENS}")
    out = args.out if args.out is not None else default_out(args.tokens)
    if not args.mog.is_file():
        raise SystemExit(f"missing model file: {args.mog}")

    prompt, _ids, expected = load_or_build_golden(
        out, args.tokens, args.model, DEFAULT_TEXT, args.dump
    )
    ensure_run(args.run)
    actual = dump_runtime_logits(args.run, args.mog, prompt)
    report(actual, expected.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()

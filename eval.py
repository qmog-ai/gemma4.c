#!/usr/bin/env python3
"""Score gemma4.c on the frozen tinyMMLU multiple-choice evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "eval" / "tiny_mmlu.json"
LABELS = "ABCD"
VOCAB_SIZE = 262_144


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summary(rows: list[dict]) -> dict:
    correct = sum(row["correct"] for row in rows)
    return {
        "completed": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "runtime_seconds": sum(row["runtime_seconds"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    model = args.model.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    for path, name in ((runtime, "runtime"), (model, "model"), (dataset, "dataset")):
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")

    questions = json.loads(dataset.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        questions = questions[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    choice_token_ids = []
    for label in LABELS:
        tokens = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(tokens) != 1:
            raise SystemExit(f"choice {label!r} is not one token: {tokens}")
        choice_token_ids.append(tokens[0])

    metadata = {
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "runtime": str(runtime),
        "runtime_sha256": sha256(runtime),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "tokenizer": args.tokenizer,
        "choice_token_ids": dict(zip(LABELS, choice_token_ids)),
        "question_count": len(questions),
    }

    rows: list[dict] = []
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("metadata") != metadata:
            raise SystemExit(f"existing output was produced by a different run: {output}")
        rows = previous.get("results", [])
        print(f"Resuming after {len(rows)} completed questions", file=sys.stderr)

    for index, question in enumerate(questions[len(rows) :], start=len(rows)):
        start = time.perf_counter()
        proc = subprocess.run(
            [str(runtime), "-m", str(model), "--dump-last-logits", question["prompt"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode:
            sys.stderr.buffer.write(proc.stderr)
            raise SystemExit(f"runtime failed on question {index} with exit code {proc.returncode}")

        logits = np.frombuffer(proc.stdout, dtype=np.float32)
        if logits.size != VOCAB_SIZE:
            raise SystemExit(
                f"question {index}: expected {VOCAB_SIZE} logits, received {logits.size}"
            )
        scores = [float(logits[token]) for token in choice_token_ids]
        prediction = int(np.argmax(scores))
        answer = int(question["answer"])
        ordered = sorted(scores, reverse=True)
        row = {
            "id": question["id"],
            "subject": question["subject"],
            "answer": answer,
            "answer_label": LABELS[answer],
            "prediction": prediction,
            "prediction_label": LABELS[prediction],
            "correct": prediction == answer,
            "choice_logits": dict(zip(LABELS, scores)),
            "choice_margin": ordered[0] - ordered[1],
            "prompt_tokens": 1 + len(tokenizer.encode(question["prompt"], add_special_tokens=False)),
            "runtime_seconds": elapsed,
        }
        rows.append(row)

        report = {"metadata": metadata, "summary": summary(rows), "results": rows}
        write_json(output, report)
        mark = "correct" if row["correct"] else "wrong"
        print(
            f"[{index + 1:3d}/{len(questions)}] {mark:7s} "
            f"{row['subject']} {row['prediction_label']}/{row['answer_label']} "
            f"{elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    result = summary(rows)
    print(f"accuracy {result['accuracy'] * 100:.1f}% ({result['correct']}/{result['completed']})")
    print(f"runtime {result['runtime_seconds']:.1f}s")
    print(f"prompt tokens {result['prompt_tokens']}")
    print(f"results {output}")


if __name__ == "__main__":
    main()

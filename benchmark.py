#!/usr/bin/env python3
"""Reproduce the gemma4-e2b.c (int8) README throughput numbers.

Methodology
-----------
gemma4-e2b.c --bench PP TG times two synthetic workloads and prints
  pp{PP} X.XX tok/s
  tg{TG} X.XX tok/s

Prefill (pp) runs one batched forward over PP dummy token ids (2, 3, ...).
Decode (tg) runs TG greedy single-token forwards, sampling argmax each step.
Neither path includes tokenization. Timing is CLOCK_MONOTONIC around the
compute only.

This script builds ./run with the Makefile default (-O3 -march=native
-fopenmp). On this Zen 4 host that enables AVX-512 VNNI. It does not set
OMP_NUM_THREADS, so gemma4-e2b.c threads() uses physical cores (8 on a 7700).

Protocol
----------------------------------
  1. One discarded warmup of --bench 512 128 (caches, page faults).
  2. Twelve measured runs of the same command.
  3. Report mean ± sample standard deviation (N-1).

Usage
-----
  python3 benchmark.py                 # README defaults: 512 / 128 / 12
  python3 benchmark.py 512 128 12      # same, explicit
  python3 benchmark.py 256 64 5        # shorter / fewer runs

Requires gemma4-E2B-int8.mog in the repo root (or MOG_FILE). Optional
RUN_BIN / MOG_FILE override the binary and checkpoint without rebuilding.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stddev(xs: list[float]) -> float:
    if len(xs) == 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_args(argv: list[str]) -> tuple[int, int, int]:
    if len(argv) > 3:
        raise SystemExit("usage: benchmark.py [prefill] [decode] [measured_runs]")
    try:
        pp = int(argv[0]) if len(argv) > 0 else 512
        tg = int(argv[1]) if len(argv) > 1 else 128
        runs = int(argv[2]) if len(argv) > 2 else 12
    except ValueError:
        raise SystemExit("usage: benchmark.py [prefill] [decode] [measured_runs]")
    if pp < 0 or tg < 0 or runs < 1:
        raise SystemExit("prefill/decode must be >= 0 and measured_runs >= 1")
    return pp, tg, runs


def bench(run: Path, mog: Path, pp: int, tg: int) -> str:
    return subprocess.check_output(
        [str(run), "-m", str(mog), "--bench", str(pp), str(tg)],
        text=True,
    )


def main() -> None:
    pp, tg, runs = parse_args(sys.argv[1:])
    run = Path(os.environ.get("RUN_BIN", ROOT / "run"))
    mog = Path(os.environ.get("MOG_FILE", ROOT / "gemma4-E2B-int8.mog"))
    if not mog.is_file():
        raise SystemExit(f"missing model file: {mog}")
    if "RUN_BIN" not in os.environ:
        subprocess.check_call(["make", "-C", str(ROOT)])

    print(f"warmup: ./run --bench {pp} {tg}")
    bench(run, mog, pp, tg)

    print(f"measured runs: {runs} (pp={pp}, tg={tg})")
    text = "".join(bench(run, mog, pp, tg) for _ in range(runs))
    pp_vals = [float(x) for x in re.findall(rf"^pp{pp}\s+([0-9.]+)\s+tok/s$", text, re.M)]
    tg_vals = [float(x) for x in re.findall(rf"^tg{tg}\s+([0-9.]+)\s+tok/s$", text, re.M)]
    if len(pp_vals) != runs or len(tg_vals) != runs:
        raise SystemExit(
            f"expected {runs} pp and tg samples, got {len(pp_vals)} pp and {len(tg_vals)} tg\n{text}"
        )

    print(
        f"prefill pp{pp}: {mean(pp_vals):.2f} ± {stddev(pp_vals):.2f} tok/s  "
        f"({', '.join(f'{v:.2f}' for v in pp_vals)})"
    )
    print(
        f"decode  tg{tg}: {mean(tg_vals):.2f} ± {stddev(tg_vals):.2f} tok/s  "
        f"({', '.join(f'{v:.2f}' for v in tg_vals)})"
    )


if __name__ == "__main__":
    main()

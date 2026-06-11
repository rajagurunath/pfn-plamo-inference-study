#!/usr/bin/env python3
"""Analyze speculative decoding benchmark results.

- Losslessness: greedy outputs of every spec config must match the baseline
  content hash for the same (task, prompt_idx).
- Summary: median tok/s per task per config, speedup vs baseline, acceptance.

Usage: python bench/analyze.py [--baseline baseline_8b]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(label):
    rows = []
    p = RESULTS / f"{label}.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_8b")
    args = ap.parse_args()

    base = load(args.baseline)
    if not base:
        raise SystemExit(f"no baseline results: {args.baseline}")
    base_hash = {(r["task"], r["prompt_idx"]): r["content_sha"] for r in base}

    labels = sorted(
        (p.stem for p in RESULTS.glob("*.jsonl") if p.stem != args.baseline),
        key=lambda s: int("".join(filter(str.isdigit, s)) or 0),
    )

    # --- losslessness check ---
    print("=== losslessness vs baseline (greedy, must match) ===")
    for label in labels:
        rows = load(label)
        mismatches = [
            (r["task"], r["prompt_idx"])
            for r in rows
            if base_hash.get((r["task"], r["prompt_idx"])) != r["content_sha"]
        ]
        uniq = sorted(set(mismatches))
        status = "OK — all outputs identical" if not uniq else f"MISMATCH at {uniq}"
        print(f"{label}: {status}")

    # --- summary table ---
    tasks = sorted({r["task"] for r in base})
    base_med = {
        t: statistics.median(r["tok_per_sec"] for r in base if r["task"] == t)
        for t in tasks
    }
    print("\n=== median decode tok/s (speedup vs baseline) [acceptance] ===")
    header = f"{'config':<14}" + "".join(f"{t:<32}" for t in tasks)
    print(header)
    row = f"{args.baseline:<14}"
    for t in tasks:
        row += f"{base_med[t]:>6.1f} tok/s (1.00x)          "
    print(row)
    for label in labels:
        rows = load(label)
        out = f"{label:<14}"
        for t in tasks:
            tr = [r for r in rows if r["task"] == t]
            if not tr:
                out += f"{'—':<32}"
                continue
            med = statistics.median(r["tok_per_sec"] for r in tr)
            dn = sum(r["draft_n"] or 0 for r in tr)
            da = sum(r["draft_n_accepted"] or 0 for r in tr)
            acc = f" [{da/dn:.0%}]" if dn else ""
            out += f"{med:>6.1f} tok/s ({med/base_med[t]:.2f}x){acc:<9}"
        print(out)


if __name__ == "__main__":
    main()

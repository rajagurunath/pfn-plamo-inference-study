#!/usr/bin/env python3
"""Speculative decoding benchmark harness for PLaMo 2 via llama-server.

Starts llama-server with a given configuration (baseline or speculative),
sends task prompts with greedy sampling, and records per-request timings
including draft acceptance stats from the server's JSON response.

Usage:
  python bench/bench.py --target gguf/plamo-2-8b-Q8_0.gguf \
      --label baseline --repeats 3
  python bench/bench.py --target gguf/plamo-2-8b-Q8_0.gguf \
      --draft gguf/plamo-2-1b-Q8_0.gguf --draft-max 8 --draft-min 1 \
      --label spec_d8 --repeats 3

Results: one JSON line per request appended to results/<label>.jsonl
"""

import argparse
import json
import hashlib
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "llama-b9596" / "llama-server"
PORT = 18080


def start_server(args):
    cmd = [
        str(SERVER_BIN),
        "-m", args.target,
        "--port", str(PORT),
        "-ngl", "99",
        "-c", str(args.ctx),
        "--no-webui",
    ]
    if args.draft:
        cmd += [
            "--spec-type", "draft-simple",
            "-md", args.draft,
            "--spec-draft-n-max", str(args.draft_max),
            "--spec-draft-n-min", str(args.draft_min),
            "--spec-draft-p-min", str(args.draft_p_min),
            "-ngld", "99",
        ]
    elif args.ngram:
        cmd += ["--spec-type", "ngram-simple"]
    log = open(ROOT / "results" / f"server-{args.label}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    # wait for health
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            sys.exit(f"server died at startup; see results/server-{args.label}.log")
        time.sleep(1)
    proc.kill()
    sys.exit("server did not become healthy in 120s")


def completion(prompt, n_predict):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--ngram", action="store_true")
    ap.add_argument("--draft-max", type=int, default=8)
    ap.add_argument("--draft-min", type=int, default=1)
    ap.add_argument("--draft-p-min", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="ja_chat,translation_en_ja,code")
    args = ap.parse_args()

    prompts = json.loads((ROOT / "bench" / "prompts.json").read_text())
    (ROOT / "results").mkdir(exist_ok=True)
    out_path = ROOT / "results" / f"{args.label}.jsonl"

    proc = start_server(args)
    print(f"server up ({args.label}); benchmarking...")
    try:
        # one warmup request (first run pays Metal shader compile etc.)
        completion("ウォームアップ。", 16)

        rows = []
        with open(out_path, "a") as f:
            for task in args.tasks.split(","):
                for pi, prompt in enumerate(prompts[task]):
                    for rep in range(args.repeats):
                        r = completion(prompt, args.n_predict)
                        t = r.get("timings", {})
                        row = {
                            "label": args.label,
                            "task": task,
                            "prompt_idx": pi,
                            "rep": rep,
                            "tok_per_sec": t.get("predicted_per_second"),
                            "n_predicted": t.get("predicted_n"),
                            "draft_n": t.get("draft_n"),
                            "draft_n_accepted": t.get("draft_n_accepted"),
                            "prompt_per_sec": t.get("prompt_per_second"),
                            "content_sha": hashlib.sha256(
                                r.get("content", "").encode()).hexdigest()[:16],
                            "content": r.get("content", ""),
                        }
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        acc = ""
                        if row["draft_n"]:
                            acc = f"  acc={row['draft_n_accepted']}/{row['draft_n']}" \
                                  f" ({row['draft_n_accepted']/row['draft_n']:.0%})"
                        print(f"{task}[{pi}] rep{rep}: "
                              f"{row['tok_per_sec']:.1f} tok/s{acc}")

        print("\n=== medians by task ===")
        for task in args.tasks.split(","):
            tr = [r for r in rows if r["task"] == task]
            med = statistics.median(r["tok_per_sec"] for r in tr)
            line = f"{task}: {med:.1f} tok/s"
            dn = sum(r["draft_n"] or 0 for r in tr)
            if dn:
                da = sum(r["draft_n_accepted"] or 0 for r in tr)
                line += f", acceptance {da/dn:.1%}"
            print(line)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()

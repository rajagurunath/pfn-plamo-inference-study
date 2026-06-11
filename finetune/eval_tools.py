#!/usr/bin/env python3
"""Tool-calling eval for plamo-3-nict-2b-base, before/after LoRA.

Conditions:
  --condition base        zero-shot base model
  --condition base-fewshot  base model with 2 in-context examples
  --condition lora        base + finetune/adapter

Metrics over data/eval.jsonl: parse rate, function-name accuracy, argument
exact-match (on call cases), false-call rate (on no-call cases).
Writes finetune/results/<condition>.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "pfnet/plamo-3-nict-2b-base"

FEWSHOT = [
    {"role": "user", "content": "What's the weather like in Tokyo today?"},
    {"role": "assistant", "content": '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'},
    {"role": "user", "content": "Thanks! And who wrote 'Kokoro'?"},
    {"role": "assistant", "content": "'Kokoro' was written by Natsume Soseki, first published in 1914."},
]

def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).split("\n\n")[0]):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["base", "base-fewshot", "lora"])
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    if args.condition == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, HERE / "adapter")
        model = model.merge_and_unload()
    model.to(device).eval()

    rows = [json.loads(l) for l in open(HERE / "data" / "eval.jsonl")]
    out_rows, t0 = [], time.time()
    for i, rec in enumerate(rows):
        msgs = rec["messages"][:-1]
        if args.condition == "base-fewshot":
            msgs = [msgs[0]] + FEWSHOT + msgs[1:]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        # stop at the template's next-turn marker if the model runs on
        text = text.split("<|plamo:tag|>")[0].strip()
        gold = rec["messages"][-1]["content"]
        out_rows.append({"kind": rec["kind"],
                         "question": rec["messages"][1]["content"],
                         "gold": gold, "pred": text})
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(rows)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # score
    call = [r for r in out_rows if r["kind"] == "call"]
    nocall = [r for r in out_rows if r["kind"] == "nocall"]
    parsed = [(r, extract_json(r["pred"])) for r in call]
    n_parse = sum(1 for _, p in parsed if isinstance(p, dict) and "name" in p)
    n_name = n_args = 0
    for r, p in parsed:
        if not (isinstance(p, dict) and "name" in p):
            continue
        g = json.loads(r["gold"])
        if p.get("name") == g.get("name"):
            n_name += 1
            if p.get("arguments") == g.get("arguments"):
                n_args += 1
    false_calls = sum(1 for r in nocall if extract_json(r["pred"]) is not None
                      and isinstance(extract_json(r["pred"]), dict)
                      and "name" in extract_json(r["pred"]))

    summary = {
        "condition": args.condition,
        "n_call": len(call), "n_nocall": len(nocall),
        "parse_rate": n_parse / len(call) if call else None,
        "func_name_acc": n_name / len(call) if call else None,
        "args_exact_match": n_args / len(call) if call else None,
        "false_call_rate": false_calls / len(nocall) if nocall else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"{args.condition}.json", "w") as f:
        json.dump({"summary": summary, "rows": out_rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

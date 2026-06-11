#!/usr/bin/env python3
"""Build a small tool-calling SFT dataset for PLaMo 3 from glaive-function-calling-v2.

Extracts single-turn examples: system tool definitions + one user query ->
assistant responds with either a JSON tool call or a plain answer (no-call).
Output: finetune/data/{train,eval}.jsonl with {"messages": [...], "kind": "call"|"nocall"}
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
random.seed(42)

N_TRAIN_CALL, N_TRAIN_NOCALL = 700, 100
N_EVAL_CALL, N_EVAL_NOCALL = 60, 20

SYSTEM_PREFIX = (
    "You are a helpful assistant with access to the following tools. "
    "When the user's request requires a tool, respond with ONLY a JSON object "
    '{"name": <function-name>, "arguments": <args-object>}. '
    "If no tool is needed, answer normally.\nTools:\n"
)

ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
print("source rows:", len(ds))

calls, nocalls = [], []
for row in ds:
    sys_raw = row["system"]
    chat = row["chat"]
    if not sys_raw.startswith("SYSTEM:"):
        continue
    # tool definitions: everything after the boilerplate sentence
    m = re.search(r"(\{.*\})", sys_raw, re.S)
    tools = m.group(1).strip() if m else None

    # first USER turn and the ASSISTANT reply that follows it
    turns = re.split(r"(USER:|ASSISTANT:)", chat)
    seq = [(turns[i], turns[i + 1].strip()) for i in range(1, len(turns) - 1, 2)]
    if len(seq) < 2 or seq[0][0] != "USER:" or seq[1][0] != "ASSISTANT:":
        continue
    user, asst = seq[0][1], seq[1][1]
    if not user or not asst or len(user) > 600:
        continue

    if asst.startswith("<functioncall>"):
        if tools is None:
            continue
        body = asst[len("<functioncall>"):].split("<|endoftext|>")[0].strip()
        # glaive quotes arguments as a string: '"arguments": '{...}'' -> normalize
        body = re.sub(r"'\s*$", "", body)
        body = body.replace("\"arguments\": '", '"arguments": ').rstrip()
        if body.endswith("'"):
            body = body[:-1]
        try:
            call = json.loads(body)
            args = call.get("arguments")
            if isinstance(args, str):
                call["arguments"] = json.loads(args)
        except Exception:
            continue
        target = json.dumps(call, ensure_ascii=False)
        calls.append((tools, user, target, "call"))
    else:
        asst = asst.split("<|endoftext|>")[0].strip()
        if tools is None or len(asst) > 400:
            continue
        nocalls.append((tools, user, asst, "nocall"))

print(f"parsed: {len(calls)} call examples, {len(nocalls)} no-call examples")
random.shuffle(calls)
random.shuffle(nocalls)

def to_rec(tools, user, target, kind):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PREFIX + tools},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "kind": kind,
    }

train = [to_rec(*x) for x in calls[:N_TRAIN_CALL] + nocalls[:N_TRAIN_NOCALL]]
eval_ = [to_rec(*x) for x in calls[N_TRAIN_CALL:N_TRAIN_CALL + N_EVAL_CALL]
         + nocalls[N_TRAIN_NOCALL:N_TRAIN_NOCALL + N_EVAL_NOCALL]]
random.shuffle(train)

for name, rows in [("train", train), ("eval", eval_)]:
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows -> {OUT}/{name}.jsonl")

#!/usr/bin/env python3
"""fig1_conditions: grouped bars — args-exact accuracy vs false-call rate across six conditions."""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "summary.json"

with open(DATA) as f:
    summary = json.load(f)

order = summary["meta"]["condition_order"]
cond = {c["id"]: c for c in summary["conditions"]}

# Short display labels in the requested order
labels = {
    "base": "base\n0-shot",
    "base-fewshot": "base\n2-shot",
    "lora-400": "LoRA v1\n400 ex",
    "lora-ckpt-step400": "LoRA v1\n3.2k ex",
    "lora-ckpt-step800": "LoRA v1\n6.4k ex",
    "lora-v2": "LoRA v2\n1,718 ex",
}

args_exact = [cond[c]["args_exact"] for c in order]
false_call = [cond[c]["false_call"] for c in order]

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(6.3, 3.6))

x = np.arange(len(order))
w = 0.38

b1 = ax.bar(x - w / 2, args_exact, w, color=BLUE, label="args-exact accuracy (40 call cases)")
b2 = ax.bar(x + w / 2, false_call, w, color=ORANGE, label="false-call rate (12 no-call cases)")

for rect, v in zip(b1, args_exact):
    ax.annotate(f"{v:.0f}" if v == int(v) else f"{v:.1f}",
                (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color=BLUE)
for i, (rect, v) in enumerate(zip(b2, false_call)):
    txt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    if order[i] == "lora-v2":
        txt += "*"
    ax.annotate(txt,
                (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color="#9C6A00")

ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in order], fontsize=9)
ax.set_ylabel("percent")
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, color="0.88", linewidth=0.8)
ax.set_axisbelow(True)

ax.set_title("Tool-calling accuracy across fine-tuning conditions", fontsize=11, pad=10)

ax.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=1.4,
          bbox_to_anchor=(0.0, 1.0))

fig.text(0.01, 0.012,
         "*LoRA v2 false-call rate measured on a stricter 12-question no-call set "
         "(other conditions share the v1 no-call set).",
         fontsize=7.5, color="0.35", ha="left")

fig.tight_layout(rect=(0, 0.045, 1, 1))

for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(HERE / f"fig1_conditions.{ext}", **kw)
print("saved fig1_conditions.{pdf,png}")

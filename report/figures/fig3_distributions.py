#!/usr/bin/env python
"""Figure 3: Per-request decode-speed distributions.

Grouped box plots: x = config (baseline, ngram, d2, d4, d8, d16),
grouped/colored by task, y = tokens/s, with individual jittered points
overlaid (12 points per config x task = 36 per config).

Reads:  report/data/summary.json
Writes: report/figures/fig3_distributions.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "summary.json"
OUT = ROOT / "figures"

# ---------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)

# Okabe-Ito (colorblind-safe)
OKABE_ITO = {
    "code": "#0072B2",              # blue
    "ja_chat": "#E69F00",           # orange
    "translation_en_ja": "#009E73", # bluish green
}
TASK_LABELS = {
    "code": "Code",
    "ja_chat": "JA chat",
    "translation_en_ja": "EN-JA translation",
}
CONFIG_LABELS = {
    "fx_baseline": "baseline",
    "fx_ngram": "ngram",
    "fx_spec_d2": "spec d2",
    "fx_spec_d4": "spec d4",
    "fx_spec_d8": "spec d8",
    "fx_spec_d16": "spec d16",
}

# ---------------------------------------------------------------- data
with open(DATA) as f:
    summary = json.load(f)

configs = summary["configs"]
tasks = summary["tasks"]
raw = summary["raw_points"]
baseline_medians = summary["baseline_medians"]

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(6.3, 3.6))

n_tasks = len(tasks)
group_width = 0.78
box_width = group_width / n_tasks * 0.82
offsets = (np.arange(n_tasks) - (n_tasks - 1) / 2) * (group_width / n_tasks)

rng = np.random.default_rng(42)

for ti, task in enumerate(tasks):
    color = OKABE_ITO[task]
    positions = np.arange(len(configs)) + offsets[ti]
    data = [raw[cfg][task] for cfg in configs]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color=color, linewidth=0.9),
        capprops=dict(color=color, linewidth=0.9),
        boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=0.9),
    )

    # jittered individual points (12 per box)
    for pos, vals in zip(positions, data):
        jitter = rng.uniform(-box_width * 0.28, box_width * 0.28, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            s=8,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
            zorder=3,
        )

# baseline median reference lines (one per task, spanning full width)
for task in tasks:
    ax.axhline(
        baseline_medians[task],
        color=OKABE_ITO[task],
        linestyle=":",
        linewidth=0.9,
        alpha=0.7,
        zorder=1,
    )

# label the baseline reference lines on the right edge
xmax = len(configs) - 0.45
for task in tasks:
    ax.annotate(
        "baseline",
        xy=(xmax, baseline_medians[task]),
        xytext=(2, 0),
        textcoords="offset points",
        fontsize=7,
        color=OKABE_ITO[task],
        va="center",
        ha="left",
        alpha=0.9,
    )

ax.set_xticks(np.arange(len(configs)))
ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
ax.set_xlabel("Configuration")
ax.set_ylabel("Decode speed (tokens/s)")
ax.set_xlim(-0.55, len(configs) - 0.25)

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO[t], alpha=0.5, edgecolor=OKABE_ITO[t])
    for t in tasks
]
ax.legend(handles, [TASK_LABELS[t] for t in tasks], loc="lower left", ncol=1)

fig.tight_layout()
fig.savefig(OUT / "fig3_distributions.pdf")
fig.savefig(OUT / "fig3_distributions.png", dpi=200)
print("saved", OUT / "fig3_distributions.pdf")
print("saved", OUT / "fig3_distributions.png")

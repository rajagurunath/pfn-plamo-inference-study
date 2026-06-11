#!/usr/bin/env python
"""Figure 2: Token acceptance rate vs. draft length.

One line per task across the speculative configs (n_max = 2, 4, 8, 16),
with the n-gram drafter shown as flat (draft-length-independent) dashed
reference markers in the same task colors. Acceptance decays with draft
depth. Palette/task colors match fig1/fig3 (Okabe-Ito).

Reads:  report/data/summary.json
Writes: report/figures/fig2_acceptance.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

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

# Okabe-Ito (colorblind-safe) -- same task colors as fig1/fig3
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
TASK_MARKERS = {
    "code": "o",
    "ja_chat": "s",
    "translation_en_ja": "^",
}

SPEC_CONFIGS = ["fx_spec_d2", "fx_spec_d4", "fx_spec_d8", "fx_spec_d16"]
DRAFT_LENGTHS = [2, 4, 8, 16]

# ---------------------------------------------------------------- data
with open(DATA) as f:
    summary = json.load(f)

tasks = summary["tasks"]
bct = summary["by_config_task"]
acc_pts = summary["acceptance_points"]

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(6.3, 3.4))

rng = np.random.default_rng(0)

for task in tasks:
    color = OKABE_ITO[task]
    marker = TASK_MARKERS[task]

    # per-request acceptance points, lightly jittered in x
    for n, cfg in zip(DRAFT_LENGTHS, SPEC_CONFIGS):
        pts = np.asarray(acc_pts[cfg][task]) * 100.0
        jitter = n * (1.0 + rng.uniform(-0.04, 0.04, size=pts.size))
        ax.scatter(
            jitter,
            pts,
            s=9,
            facecolor=color,
            edgecolor="none",
            alpha=0.30,
            zorder=2,
        )

    # aggregate acceptance line over draft length
    y = [bct[cfg][task]["acceptance_rate"] * 100.0 for cfg in SPEC_CONFIGS]
    ax.plot(
        DRAFT_LENGTHS,
        y,
        color=color,
        marker=marker,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=1.4,
        zorder=3,
    )

    # n-gram drafter: flat (no draft-length sweep) dashed reference
    ng = bct["fx_ngram"][task]["acceptance_rate"] * 100.0
    ax.axhline(ng, color=color, linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.7, zorder=1)
    ax.plot(
        DRAFT_LENGTHS,
        [ng] * len(DRAFT_LENGTHS),
        linestyle="none",
        marker=marker,
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=3,
    )

# ---------------------------------------------------------------- axes
ax.set_xscale("log", base=2)
ax.set_xticks(DRAFT_LENGTHS)
ax.set_xticklabels([str(n) for n in DRAFT_LENGTHS])
ax.minorticks_off()
ax.set_xlabel("Draft length n_max")
ax.set_ylabel("Token acceptance rate")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_xlim(1.75, 18.5)

# ---------------------------------------------------------------- legend
task_handles = [
    Line2D(
        [], [],
        color=OKABE_ITO[t],
        marker=TASK_MARKERS[t],
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=1.4,
        label=TASK_LABELS[t],
    )
    for t in tasks
]
style_handles = [
    Line2D([], [], color="0.35", marker="o", markersize=5,
           markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
           label="Draft model (n_max sweep)"),
    Line2D([], [], color="0.35", linestyle=(0, (4, 3)), linewidth=0.9,
           marker="o", markersize=4.5, markerfacecolor="white",
           markeredgecolor="0.35", markeredgewidth=1.0,
           label="n-gram drafter (flat)"),
]
leg1 = ax.legend(handles=task_handles, loc="upper right", handlelength=1.8,
                 borderaxespad=0.2)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc="lower left", handlelength=2.2,
          borderaxespad=0.2)

fig.tight_layout()

# ---------------------------------------------------------------- save
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig2_acceptance.pdf")
fig.savefig(OUT / "fig2_acceptance.png", dpi=200)
print("wrote", OUT / "fig2_acceptance.pdf")
print("wrote", OUT / "fig2_acceptance.png")

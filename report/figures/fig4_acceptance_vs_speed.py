#!/usr/bin/env python
"""Fig 4: per-request token acceptance rate vs per-request speedup over the
task baseline median. Colored by task, marker shape by config.

The figure carries the overhead argument: even requests with very high
acceptance (fx_spec_d2, ~0.9) sit below the 1.0x baseline line.

Reads:  report/data/summary.json
Writes: report/figures/fig4_acceptance_vs_speed.{pdf,png}
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "summary.json")

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}

TASK_COLORS = {
    "code": OKABE_ITO["blue"],
    "ja_chat": OKABE_ITO["vermillion"],
    "translation_en_ja": OKABE_ITO["green"],
}
TASK_LABELS = {
    "code": "Code",
    "ja_chat": "Japanese chat",
    "translation_en_ja": "Translation EN-JA",
}

CONFIG_MARKERS = {
    "fx_ngram": "o",
    "fx_spec_d2": "s",
    "fx_spec_d4": "^",
    "fx_spec_d8": "D",
    "fx_spec_d16": "v",
}
CONFIG_LABELS = {
    "fx_ngram": "n-gram",
    "fx_spec_d2": "spec $n_{max}$=2",
    "fx_spec_d4": "spec $n_{max}$=4",
    "fx_spec_d8": "spec $n_{max}$=8",
    "fx_spec_d16": "spec $n_{max}$=16",
}

# ---------------------------------------------------------------- data
with open(DATA) as f:
    summary = json.load(f)

baseline = summary["baseline_medians"]

fig, ax = plt.subplots(figsize=(6.3, 4.2))

for config, marker in CONFIG_MARKERS.items():
    for task, color in TASK_COLORS.items():
        acc = summary["acceptance_points"][config][task]
        tps = summary["raw_points"][config][task]
        if not acc:
            continue
        # acceptance_points may be shorter than raw_points when some
        # requests produced no draft tokens (e.g. n-gram); pair the
        # available per-request values.
        n = min(len(acc), len(tps))
        speedup = [v / baseline[task] for v in tps[:n]]
        ax.scatter(acc[:n], speedup, marker=marker, s=28,
                   facecolors=color, edgecolors="white",
                   linewidths=0.4, alpha=0.85, zorder=3)

# Baseline reference line
ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--", zorder=2)
ax.text(0.015, 1.0, "baseline (1.0x)", va="bottom", ha="left",
        fontsize=8.5, color="0.35",
        transform=ax.get_yaxis_transform())

# Annotate the overhead argument: the fx_spec_d2 cluster has ~0.9
# acceptance but still sits below 1.0x.
ax.annotate("high acceptance,\nstill slower",
            xy=(0.885, 0.76), xycoords="data",
            xytext=(0.80, 0.22), textcoords="data",
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="0.25", linewidth=0.8))

ax.set_xlabel("Token acceptance rate")
ax.set_ylabel("Speedup vs baseline (x)")
ax.set_xlim(0, 1.0)

# Legends: colors = task, marker shapes = config
task_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=6,
           markerfacecolor=TASK_COLORS[t], markeredgecolor="white",
           markeredgewidth=0.4, label=TASK_LABELS[t])
    for t in summary["tasks"]
]
config_handles = [
    Line2D([], [], marker=CONFIG_MARKERS[c], linestyle="none",
           markersize=6, markerfacecolor="0.45", markeredgecolor="white",
           markeredgewidth=0.4, label=CONFIG_LABELS[c])
    for c in CONFIG_MARKERS
]
leg1 = ax.legend(handles=task_handles, title="Task",
                 loc="upper left", bbox_to_anchor=(1.01, 1.0),
                 alignment="left")
leg1.get_title().set_fontsize(9)
ax.add_artist(leg1)
leg2 = ax.legend(handles=config_handles, title="Config",
                 loc="upper left", bbox_to_anchor=(1.01, 0.60),
                 alignment="left")
leg2.get_title().set_fontsize(9)

# Reserve room on the right for the two outside legends
# (tight_layout ignores legends placed outside the axes).
fig.subplots_adjust(left=0.095, right=0.745, top=0.97, bottom=0.115)

for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(os.path.join(HERE, f"fig4_acceptance_vs_speed.{ext}"), **kw)
print("saved fig4_acceptance_vs_speed.{pdf,png}")

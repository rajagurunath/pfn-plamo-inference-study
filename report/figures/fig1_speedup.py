#!/usr/bin/env python
"""Figure 1: decode speedup vs. draft length, per task.

Headline figure: speculative decoding with the fixed-draft model is below
the autoregressive baseline (speedup < 1.0) for nearly every configuration;
the only points above 1.0 are n-gram drafting and d4 on EN->JA translation.

Reads:  report/data/summary.json
Writes: report/figures/fig1_speedup.{pdf,png}
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "summary.json")

# --- Style: research-grade, serif, colorblind-safe (Okabe-Ito) ----------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

# Okabe-Ito palette
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "black": "#000000",
}

TASK_STYLE = {
    "code": (OKABE_ITO["blue"], "Code"),
    "ja_chat": (OKABE_ITO["vermillion"], "JA chat"),
    "translation_en_ja": (OKABE_ITO["green"], "Translation EN" + "→" + "JA"),
}

SPEC_CONFIGS = ["fx_spec_d2", "fx_spec_d4", "fx_spec_d8", "fx_spec_d16"]
DRAFT_N = [2, 4, 8, 16]


def main() -> None:
    with open(DATA) as f:
        summary = json.load(f)
    by_ct = summary["by_config_task"]

    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    # x positions: log2(n_max) for speculative configs; n-gram is a
    # separate categorical slot at the left edge.
    spec_x = [math.log2(n) for n in DRAFT_N]  # 1, 2, 3, 4
    ngram_x = 0.0

    for task, (color, label) in TASK_STYLE.items():
        y = [by_ct[c][task]["speedup_vs_baseline"] for c in SPEC_CONFIGS]
        ax.plot(
            spec_x,
            y,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.4,
            label=label,
            zorder=3,
        )
        # n-gram drafting: task-colored star at the left categorical slot
        y_ngram = by_ct["fx_ngram"][task]["speedup_vs_baseline"]
        ax.plot(
            ngram_x,
            y_ngram,
            color=color,
            marker="*",
            markersize=13,
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=4,
        )

    # Baseline reference line at speedup = 1.0
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        4.38,
        1.025,
        "baseline",
        va="bottom",
        ha="right",
        fontsize=9,
        color="0.35",
    )

    # Light separator between the n-gram category and the n_max axis
    ax.axvline(0.5, color="0.8", linewidth=0.8, zorder=1)

    # Annotate the two points that beat the baseline
    y_ng_tr = by_ct["fx_ngram"]["translation_en_ja"]["speedup_vs_baseline"]
    y_d4_tr = by_ct["fx_spec_d4"]["translation_en_ja"]["speedup_vs_baseline"]
    ax.annotate(
        f"{y_ng_tr:.2f}" + "×",
        xy=(ngram_x, y_ng_tr),
        xytext=(ngram_x + 0.12, y_ng_tr + 0.07),
        fontsize=8,
        color=OKABE_ITO["green"],
    )
    ax.annotate(
        f"{y_d4_tr:.2f}" + "×",
        xy=(2.0, y_d4_tr),
        xytext=(2.08, y_d4_tr + 0.06),
        fontsize=8,
        color=OKABE_ITO["green"],
    )

    ax.set_xticks([ngram_x] + spec_x)
    ax.set_xticklabels(["n-gram", "2", "4", "8", "16"])
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(0.0, 1.45)
    ax.set_xlabel("Draft length n_max (log scale)  |  n-gram drafting at left")
    ax.set_ylabel("Speedup vs. baseline (" + "×" + ")")

    # Legend: task colors plus marker semantics
    handles, labels = ax.get_legend_handles_labels()
    star = plt.Line2D(
        [],
        [],
        color="0.3",
        marker="*",
        markersize=11,
        linestyle="none",
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="n-gram drafting",
    )
    circ = plt.Line2D(
        [],
        [],
        color="0.3",
        marker="o",
        markersize=5,
        linestyle="-",
        linewidth=1.2,
        label="Draft model (spec.)",
    )
    ax.legend(
        handles=handles + [circ, star],
        loc="lower left",
        bbox_to_anchor=(0.015, 0.03),
        ncol=1,
        handlelength=1.8,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_speedup.pdf"))
    fig.savefig(os.path.join(HERE, "fig1_speedup.png"), dpi=200)
    print("wrote fig1_speedup.pdf and fig1_speedup.png")


if __name__ == "__main__":
    main()

"""Fig 2: LoRA training loss curves (v1 halted run vs v2 finished run)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "summary.json")

# --- style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
})

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"


def ema(y, alpha=0.15):
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


with open(DATA) as f:
    lc = json.load(f)["loss_curves"]

v1 = lc["v1_n11182"]
v2 = lc["v2_n1716"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=False)

# --- panel 1: v1 (KILLED) ---------------------------------------------------
s1, l1 = np.array(v1["steps"]), np.array(v1["loss"])
ax1.plot(s1, l1, color=VERMILION, lw=0.8, alpha=0.25)
ax1.plot(s1, ema(l1), color=VERMILION, lw=1.6,
         label="v1 (11,182 ex) — halted")
ax1.set_title("v1 run: halted", fontsize=10)
ax1.set_xlabel("optimizer step")
ax1.set_ylabel("training loss")
ax1.axvline(s1[-1], color="#555555", lw=0.8, ls=":")
ax1.annotate("halted at step ~870\n(label flaw found)",
             xy=(s1[-1], l1[-1]), xytext=(0.50, 0.55),
             textcoords="axes fraction", fontsize=8.5, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))
ax1.set_xlim(0, 960)

# --- panel 2: v2 (FINISHED) --------------------------------------------------
s2, l2 = np.array(v2["steps"]), np.array(v2["loss"])
ax2.plot(s2, l2, color=BLUE, lw=0.8, alpha=0.25)
ax2.plot(s2, ema(l2), color=BLUE, lw=1.6,
         label="v2 (1,716 ex) — finished")
ax2.set_title("v2 run: finished", fontsize=10)
ax2.set_xlabel("optimizer step")
ax2.annotate("complete (215 steps)\nfinal loss 0.27",
             xy=(s2[-1], l2[-1]), xytext=(0.55, 0.55),
             textcoords="axes fraction", fontsize=8.5, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))
ax2.set_xlim(0, 230)

for ax in (ax1, ax2):
    ax.set_ylim(bottom=0)

fig.suptitle("LoRA fine-tuning loss: v1 (halted, flawed labels) vs v2 (fixed data)",
             fontsize=10.5, y=1.02)
fig.tight_layout()

for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(os.path.join(HERE, f"fig2_losscurves.{ext}"),
                bbox_inches="tight", **kw)
print("saved fig2_losscurves.pdf/.png")

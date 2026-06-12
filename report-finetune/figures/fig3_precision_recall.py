"""fig3_precision_recall: call recall (parse rate) vs precision of emitted calls
for the four LoRA points, plotted against examples seen (log x).

Story: precision of emitted calls stays at/near 100% throughout, while call
recall collapses along the v1 run (92.5 -> 87.5 -> 55.0) and is restored by
v2's fixed data (100). "The model never lost the ability to call -- it lost
the will."
"""

import json
import matplotlib.pyplot as plt

DATA = "/Users/gurunathlunkupalivenugopal/ionet/pfn-try/report-finetune/data/summary.json"
OUT = "/Users/gurunathlunkupalivenugopal/ionet/pfn-try/report-finetune/figures/fig3_precision_recall"

with open(DATA) as f:
    summary = json.load(f)

cond = {c["id"]: c for c in summary["conditions"]}

# x = examples seen; recall = parse rate (share of call cases answered with a call)
# precision of emitted calls: counts verified upstream (sprint 36/37, 3.2k 35/35,
# 6.4k 22/22, v2 40/40)
points = [
    # id, examples, precision %, label, counts
    ("lora-400",          400,  100 * 36 / 37, "v1 sprint\n400 ex",  "36/37"),
    ("lora-v2",           1718, 100 * 40 / 40, "v2 fixed data\n1,718 ex", "40/40"),
    ("lora-ckpt-step400", 3200, 100 * 35 / 35, "v1 ckpt\n3.2k ex",   "35/35"),
    ("lora-ckpt-step800", 6400, 100 * 22 / 22, "v1 ckpt\n6.4k ex",   "22/22"),
]
recall = {pid: cond[pid]["parse"] for pid, *_ in points}

# Okabe-Ito
BLUE = "#0072B2"
VERMILLION = "#D55E00"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(6.3, 3.8))

# --- v1 trajectory (connected) and v2 (separate marker, different run) ---
v1_ids = ["lora-400", "lora-ckpt-step400", "lora-ckpt-step800"]
v1_x = [400, 3200, 6400]
v2_x = 1718

v1_prec = [next(p[2] for p in points if p[0] == i) for i in v1_ids]
v1_rec = [recall[i] for i in v1_ids]
v2_prec = next(p[2] for p in points if p[0] == "lora-v2")
v2_rec = recall["lora-v2"]

# precision of emitted calls
ax.plot(v1_x, v1_prec, "-o", color=BLUE, lw=1.8, ms=6, zorder=3,
        label="precision of emitted calls (v1 run)")
# v2 has recall = precision = 100, so the two stars coincide:
# draw a larger blue star with a smaller vermillion star nested inside
ax.plot([v2_x], [v2_prec], marker="*", color=BLUE, ms=19, ls="none", zorder=4,
        label="precision, v2 (fixed data)")

# call recall
ax.plot(v1_x, v1_rec, "-o", color=VERMILLION, lw=1.8, ms=6, zorder=3,
        label="call recall = parse rate (v1 run)")
ax.plot([v2_x], [v2_rec], marker="*", color=VERMILLION, ms=9, ls="none", zorder=5,
        label="call recall, v2 (fixed data)")

# --- point annotations ---
ann = dict(fontsize=8.5, ha="center")
ax.annotate("92.5", (400, 92.5), xytext=(0, -14), textcoords="offset points",
            color=VERMILLION, **ann)
ax.annotate("87.5", (3200, 87.5), xytext=(0, -14), textcoords="offset points",
            color=VERMILLION, **ann)
ax.annotate("55.0", (6400, 55.0), xytext=(-16, -3), textcoords="offset points",
            color=VERMILLION, **ann)
ax.annotate("v2: recall 100, precision 100 (40/40)", (1718, 100.0),
            xytext=(0, 9), textcoords="offset points",
            color="#333333", **ann)
ax.annotate("97.3 (36/37)", (400, 97.3), xytext=(6, 8), textcoords="offset points",
            color=BLUE, ha="left", fontsize=8.5)
ax.annotate("100 (35/35)", (3200, 100.0), xytext=(0, -16), textcoords="offset points",
            color=BLUE, **ann)
ax.annotate("100 (22/22)", (6400, 100.0), xytext=(0, 7), textcoords="offset points",
            color=BLUE, **ann)

# the collapse arrow + takeaway
ax.annotate(
    "when it does call, it is almost\nnever wrong — it just stops calling",
    xy=(6400, 57), xytext=(1450, 62),
    fontsize=9, style="italic", color="#333333", ha="center",
    arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0),
)

ax.set_xscale("log")
ax.set_xticks([400, 1718, 3200, 6400])
ax.set_xticklabels(["400", "1,718\n(v2)", "3,200", "6,400"])
ax.minorticks_off()
ax.set_xlabel("training examples seen")
ax.set_ylabel("rate (%)")
ax.set_ylim(45, 108)
ax.set_xlim(330, 8200)
ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
ax.set_axisbelow(True)

ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=1,
          handletextpad=0.6, borderaxespad=0.2)

fig.tight_layout()
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=200)
print("saved", OUT + ".{pdf,png}")

"""
make_fig3_auc.py

Generates Figure 3: predictive performance comparison bar chart
(baseline geometry vs. topology-only vs. combined), with 95% CI
error bars and a chance-level reference line.

Run analysis.py first (or supply your own AUC values) to get the
numbers used here; the values below are the ones reported in the
published paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 10.5,
    "axes.linewidth": 0.7, "axes.edgecolor": "#444", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444",
    "figure.dpi": 200,
})

NAVY = "#2f5c8f"
labels = ["Baseline\ngeometry", "Topology\nonly", "Baseline +\ntopology"]
auc = [0.6811, 0.5482, 0.6742]
lo = [0.677, 0.510, 0.667]
hi = [0.686, 0.587, 0.682]
err = [[a - l for a, l in zip(auc, lo)], [h - a for a, h in zip(auc, hi)]]
cols = [NAVY, "#c08a3e", "#5a3d8f"]

fig, ax = plt.subplots(figsize=(4.0, 3.5))
x = np.arange(3)

yticks = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
for yt in yticks:
    ax.axhline(yt, color="#ececec", lw=0.7, zorder=0)

ax.bar(x, auc, width=0.6, color=cols, edgecolor="white", linewidth=0.8, zorder=2)
ax.errorbar(x, auc, yerr=err, fmt="none", ecolor="#333", elinewidth=1.2,
            capsize=5, capthick=1.2, zorder=3)
chance_line = ax.axhline(0.5, color="#888", ls="--", lw=1.1, zorder=4)

pad = 0.012
for xi, a, h in zip(x, auc, hi):
    ax.text(xi, h + pad, f"{a:.3f}", ha="center", va="bottom", fontsize=10, zorder=5)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(yticks)
ax.set_ylim(0.40, 0.78)
ax.set_ylabel("Cross-validated AUC")
ax.set_title("Predictive performance comparison\n(5-fold CV, 95% CI)", fontsize=11, pad=8)

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=3)
ax.set_axisbelow(True)

chance = Line2D([0], [0], color="#888", ls="--", lw=1.1, label="Chance (AUC = 0.5)")
ax.legend(handles=[chance], loc="upper center", frameon=False, fontsize=9,
          bbox_to_anchor=(0.5, 1.005), handlelength=1.8)

fig.tight_layout()
fig.savefig("fig3_auc.pdf", bbox_inches="tight")
print("saved fig3_auc.pdf")

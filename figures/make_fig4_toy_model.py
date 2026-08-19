"""
make_fig4_toy_model.py

Generates Figure 4: max persistence vs. convex hull area scatter
plot from the Appendix A.1 synthetic simulation. Run
appendix/toy_model.py first to produce max_persist.npy,
hull_areas.npy, and n_loops.npy in the same directory (or point the
paths below at wherever you saved them).
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 10.5,
    "axes.linewidth": 0.7, "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444", "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "figure.dpi": 200,
})

NAVY = "#2f5c8f"
CRIM = "#a83246"

max_persist = np.load("../appendix/max_persist.npy")
hull_areas = np.load("../appendix/hull_areas.npy")
n_loops = np.load("../appendix/n_loops.npy")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
zero_mask = n_loops == 0
one_mask = n_loops == 1
more_mask = n_loops > 1

ax.scatter(hull_areas[zero_mask], max_persist[zero_mask], s=18, color="#b0b0b0",
           alpha=0.55, label="0 loops", zorder=2, edgecolor="none")
ax.scatter(hull_areas[one_mask], max_persist[one_mask], s=20, color=NAVY,
           alpha=0.65, label="1 loop", zorder=3, edgecolor="none")
ax.scatter(hull_areas[more_mask], max_persist[more_mask], s=28, color=CRIM,
           alpha=0.85, label="> 1 loop", zorder=4, edgecolor="white", linewidth=0.4)

ax.set_xlabel("Convex hull area")
ax.set_ylabel("Max persistence ($H_1$)")
ax.set_title("Synthetic 8-point clouds ($n=1{,}000$ trials)\nmax persistence vs. hull area", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=3)
fig.tight_layout()
fig.savefig("fig4_toy_model.pdf", bbox_inches="tight")
print("saved fig4_toy_model.pdf")

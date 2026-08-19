"""
toy_model.py

Appendix A.1: Synthetic validation of the sparsity argument.

Generates 1,000 independent 8-point clouds, each drawn uniformly at
random from a square region, and computes both the max persistence
(H1) and the convex hull area for each. Reports the fraction of
clouds with 0 / 1 / >1 topological features, and the correlation
between max persistence and hull area.
"""
import numpy as np
from scipy.spatial import ConvexHull
from ripser import ripser
from scipy.stats import spearmanr
import json

np.random.seed(42)
N_TRIALS = 1000
N_POINTS = 8
FIELD_SIZE = 20.0  # arbitrary square region, uniform random points

max_persist = []
n_loops = []
hull_areas = []

for i in range(N_TRIALS):
    pts = np.random.uniform(0, FIELD_SIZE, size=(N_POINTS, 2))
    result = ripser(pts, maxdim=1)
    h1 = result['dgms'][1]
    if len(h1) == 0:
        mp = 0.0
        nloops = 0
    else:
        lifetimes = h1[:, 1] - h1[:, 0]
        mp = lifetimes.max()
        nloops = len(h1)
    max_persist.append(mp)
    n_loops.append(nloops)

    try:
        hull = ConvexHull(pts)
        area = hull.volume  # 'volume' is area in 2D for scipy ConvexHull
    except Exception:
        area = np.nan
    hull_areas.append(area)

max_persist = np.array(max_persist)
n_loops = np.array(n_loops)
hull_areas = np.array(hull_areas)

frac_zero = np.mean(n_loops == 0)
frac_one = np.mean(n_loops == 1)
frac_more = np.mean(n_loops > 1)

mask = max_persist > 0
pearson_all = np.corrcoef(max_persist, hull_areas)[0, 1]
pearson_nonzero = np.corrcoef(max_persist[mask], hull_areas[mask])[0, 1]
spearman_all = spearmanr(max_persist, hull_areas).correlation
spearman_nonzero = spearmanr(max_persist[mask], hull_areas[mask]).correlation

summary = {
    "n_trials": N_TRIALS,
    "n_points": N_POINTS,
    "frac_zero_loops": frac_zero,
    "frac_one_loop": frac_one,
    "frac_more_than_one_loop": frac_more,
    "max_loops_observed": int(n_loops.max()),
    "pearson_all": pearson_all,
    "pearson_nonzero_only": pearson_nonzero,
    "spearman_all": spearman_all,
    "spearman_nonzero_only": spearman_nonzero,
    "n_nonzero": int(mask.sum()),
}
print(json.dumps(summary, indent=2))

np.save("max_persist.npy", max_persist)
np.save("hull_areas.npy", hull_areas)
np.save("n_loops.npy", n_loops)

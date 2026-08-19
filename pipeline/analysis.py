"""
analysis.py

Reproduces every statistical result reported in the paper from
combined_def.csv / combined_all22.csv:
  - Table 1: individual feature comparisons (t-test, Mann-Whitney, Cohen's d)
  - Table 2: redundancy analysis (R^2 of geometry -> topology, 5-fold CV)
  - Table 3: predictive model comparison (5-fold CV AUC)
  - Table 4: hard-case subset analysis
  - Section 3.3: all-22 supplementary check
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('combined_def.csv')
all22 = pd.read_csv('combined_all22.csv')

topo_feats = ['max_pers', 'tot_pers', 'nfeat', 'entropy']
geom_feats = ['hull_area', 'nearest_def_dist', 'spread']
labels = {
    'max_pers': 'Max persistence', 'tot_pers': 'Total persistence',
    'nfeat': 'Number of features', 'entropy': 'Persistence entropy',
    'hull_area': 'Defender hull area', 'nearest_def_dist': 'Nearest-defender distance',
    'spread': 'Defender spread',
}

# ============================================================
# TABLE 1: Individual feature comparisons
# ============================================================
print("=" * 70)
print("TABLE 1: Individual feature comparisons")
print("=" * 70)
bd = df[df.outcome == 'breakdown']
cs = df[df.outcome == 'clean_stop']
for f in topo_feats + geom_feats:
    b, c = bd[f].dropna(), cs[f].dropna()
    t, tp = stats.ttest_ind(b, c)
    u, up = stats.mannwhitneyu(b, c, alternative='greater')
    print(f"{labels[f]:28s} BD={b.mean():8.3f} ({b.std():.2f})  "
          f"CS={c.mean():8.3f} ({c.std():.2f})  t-p={tp:.4f}  MW-p={up:.4f}")

d_hull = (bd['hull_area'].mean() - cs['hull_area'].mean()) / np.sqrt(
    (bd['hull_area'].std()**2 + cs['hull_area'].std()**2) / 2)
d_maxp = (bd['max_pers'].mean() - cs['max_pers'].mean()) / np.sqrt(
    (bd['max_pers'].std()**2 + cs['max_pers'].std()**2) / 2)
print(f"\nCohen's d, hull area: {d_hull:.3f}")
print(f"Cohen's d, max persistence: {d_maxp:.3f}")

# ============================================================
# TABLE 2: Redundancy analysis
# ============================================================
print("\n" + "=" * 70)
print("TABLE 2: Redundancy analysis (R^2, geometry -> topology, 5-fold CV)")
print("=" * 70)
df_clean = df.dropna(subset=topo_feats + geom_feats)
X_geom = df_clean[geom_feats].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tf in topo_feats:
    y = df_clean[tf].values
    r2s = []
    for train_idx, test_idx in kf.split(X_geom):
        reg = LinearRegression().fit(X_geom[train_idx], y[train_idx])
        pred = reg.predict(X_geom[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    print(f"{labels[tf]:22s} R2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

print("\nPearson correlations (topology vs. each geometric feature):")
for tf in topo_feats:
    for gf in geom_feats:
        r = np.corrcoef(df_clean[tf], df_clean[gf])[0, 1]
        print(f"  {tf:12s} vs {gf:20s} r={r:.3f}")

# ============================================================
# TABLE 3: Predictive model comparison
# ============================================================
print("\n" + "=" * 70)
print("TABLE 3: 5-fold CV AUC")
print("=" * 70)
df_clean = df_clean.copy()
df_clean['y'] = (df_clean['outcome'] == 'breakdown').astype(int)


def cv_auc(feats, seed=42):
    X = df_clean[feats].values
    y = df_clean['y'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    oof_pred = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        oof_pred[test_idx] = pred
        aucs.append(roc_auc_score(y[test_idx], pred))
    return np.array(aucs), oof_pred


baseline_aucs, baseline_oof = cv_auc(geom_feats)
topo_aucs, _ = cv_auc(topo_feats)
combined_aucs, _ = cv_auc(geom_feats + topo_feats)

print(f"Baseline geometry   AUC = {baseline_aucs.mean():.4f}  SD = {baseline_aucs.std():.4f}")
print(f"Topology only       AUC = {topo_aucs.mean():.4f}  SD = {topo_aucs.std():.4f}")
print(f"Baseline + topology AUC = {combined_aucs.mean():.4f}  SD = {combined_aucs.std():.4f}")

tstat, pval = stats.ttest_rel(combined_aucs, baseline_aucs)
print(f"\nPaired t-test (combined vs baseline): "
      f"delta={combined_aucs.mean() - baseline_aucs.mean():.4f}, p={pval:.4f}")

df_clean['baseline_oof'] = baseline_oof

# ============================================================
# TABLE 4: Hard-case subset
# ============================================================
print("\n" + "=" * 70)
print("TABLE 4: Hard-case subset (bottom 40% by baseline prediction uncertainty)")
print("=" * 70)
df_clean['dist_to_half'] = (df_clean['baseline_oof'] - 0.5).abs()
hard = df_clean.nsmallest(int(len(df_clean) * 0.4), 'dist_to_half').copy()
print(f"Hard-case subset size: {len(hard)}")

bdh = hard[hard.outcome == 'breakdown']
csh = hard[hard.outcome == 'clean_stop']
for f in ['max_pers', 'tot_pers', 'nfeat']:
    t, tp = stats.ttest_ind(bdh[f].dropna(), csh[f].dropna())
    print(f"  {labels[f]:22s} BD={bdh[f].mean():.3f}  CS={csh[f].mean():.3f}  t-p={tp:.3f}")


def cv_auc_subset(sub, feats, seed=42):
    X = sub[feats].values
    y = sub['y'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        aucs.append(roc_auc_score(y[test_idx], pred))
    return np.array(aucs)


base_h = cv_auc_subset(hard, geom_feats)
comb_h = cv_auc_subset(hard, geom_feats + topo_feats)
print(f"\nHard-case baseline AUC = {base_h.mean():.3f}")
print(f"Hard-case combined AUC = {comb_h.mean():.3f}")
t, p = stats.ttest_rel(comb_h, base_h)
print(f"delta = {comb_h.mean() - base_h.mean():.3f}, p={p:.3f}")

# ============================================================
# Section 3.3: All-22 supplementary check
# ============================================================
print("\n" + "=" * 70)
print("ALL-22 SUPPLEMENTARY CHECK")
print("=" * 70)
bd22 = all22[all22.outcome == 'breakdown']
cs22 = all22[all22.outcome == 'clean_stop']
for f in ['max_pers_22', 'tot_pers_22', 'nfeat_22', 'entropy_22']:
    b, c = bd22[f].dropna(), cs22[f].dropna()
    t, tp = stats.ttest_ind(b, c)
    print(f"  {f:14s} BD={b.mean():.3f}  CS={c.mean():.3f}  t-p={tp:.3f}")
print(f"\navg n_points in all-22 cloud: {all22['n_points_22'].mean():.2f}")

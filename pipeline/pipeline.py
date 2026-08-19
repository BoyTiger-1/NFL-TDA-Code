"""
pipeline.py

Builds defender-only point clouds for the 6 frames preceding each play's
pass_forward event, computes persistent homology features (via ripser)
and geometric baseline features, for a balanced sample of completed
passes (breakdown vs. clean stop) drawn from each week of the 2018
NFL season (NFL Big Data Bowl 2021 public dataset).

Usage:
    python3 pipeline.py <week_number>

Writes week{N}_def.csv (defender-only features) and week{N}_all22.csv
(all-22 supplementary features) to the current directory.
"""
import pandas as pd
import numpy as np
from ripser import ripser
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, cdist
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/mnt/user-data/uploads'

DEF_POS = {'CB', 'SS', 'FS', 'MLB', 'OLB', 'ILB', 'LB', 'DB', 'NT', 'DE', 'DL', 'S'}
OFF_SKILL_POS = {'QB', 'WR', 'TE', 'RB', 'FB', 'HB'}


def compute_frame_topo(pts):
    """pts: Nx2 array of (x, y) positions for one frame.
    Returns (max_persistence, total_persistence, num_features, persistence_entropy)
    computed from the dimension-1 persistence diagram of the Vietoris-Rips
    filtration on pts."""
    if len(pts) < 3:
        return np.nan, np.nan, np.nan, np.nan
    res = ripser(pts, maxdim=1)
    h1 = res['dgms'][1]
    if len(h1) == 0:
        return 0.0, 0.0, 0, 0.0
    life = h1[:, 1] - h1[:, 0]
    life = life[np.isfinite(life)]
    if len(life) == 0:
        return 0.0, 0.0, 0, 0.0
    p = life / life.sum()
    ent = -(p * np.log(p)).sum()
    return life.max(), life.sum(), len(life), ent


def compute_frame_geom(def_pts, off_pts):
    """def_pts: defender Nx2 positions for one frame.
    off_pts: offensive-skill-player Nx2 positions for the same frame
    (used only for the mean nearest-defender-distance feature).
    Returns (hull_area, mean_defender_spread, mean_nearest_defender_distance)."""
    if len(def_pts) < 3:
        hull_area = np.nan
    else:
        try:
            hull_area = ConvexHull(def_pts).volume  # 'volume' = area in 2D
        except Exception:
            hull_area = np.nan
    if len(def_pts) >= 2:
        spread = pdist(def_pts).mean()
    else:
        spread = np.nan
    if len(off_pts) > 0 and len(def_pts) > 0:
        dmat = cdist(off_pts, def_pts)
        nearest = dmat.min(axis=1).mean()
    else:
        nearest = np.nan
    return hull_area, spread, nearest


def process_week(week_num, sample_plays_df):
    """
    sample_plays_df: DataFrame with columns gameId, playId, outcome
    for the specific plays sampled for this week (40 breakdown, 40
    clean stop, or as many as were available).

    Returns (def_features_df, all22_features_df).
    """
    keys = set(zip(sample_plays_df.gameId, sample_plays_df.playId))
    usecols = ['gameId', 'playId', 'frameId', 'event', 'nflId', 'position', 'x', 'y']

    chunks = []
    for chunk in pd.read_csv(f'{DATA_DIR}/week{week_num}.csv', usecols=usecols, chunksize=1_000_000):
        sub = chunk[chunk.set_index(['gameId', 'playId']).index.isin(keys)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        return None, None
    df = pd.concat(chunks, ignore_index=True)

    # find pass_forward frame per play (first occurrence)
    pf = df[df['event'] == 'pass_forward'].groupby(['gameId', 'playId'])['frameId'].min().reset_index()
    pf = pf.rename(columns={'frameId': 'pf_frame'})
    df = df.merge(pf, on=['gameId', 'playId'], how='inner')  # drop plays w/o pass_forward event found

    # six frames immediately preceding pass_forward (not including pass_forward itself)
    df = df[(df['frameId'] >= df['pf_frame'] - 6) & (df['frameId'] <= df['pf_frame'] - 1)]

    def_rows = df[df['position'].isin(DEF_POS)]
    off_rows = df[df['position'].isin(OFF_SKILL_POS)]

    def_results = []
    all22_results = []
    for (gid, pid), g in def_rows.groupby(['gameId', 'playId']):
        frame_feats = []
        frame_feats_all22 = []
        off_g = off_rows[(off_rows.gameId == gid) & (off_rows.playId == pid)]
        for fid, fg in g.groupby('frameId'):
            pts = fg[['x', 'y']].values
            mp, tp, nf, ent = compute_frame_topo(pts)
            off_fg = off_g[off_g.frameId == fid]
            ha, sp, nd = compute_frame_geom(pts, off_fg[['x', 'y']].values)
            frame_feats.append((mp, tp, nf, ent, ha, sp, nd, len(pts)))

            # all-22 (defenders + offensive skill players in same frame)
            all_pts = np.vstack([pts, off_fg[['x', 'y']].values]) if len(off_fg) else pts
            mp2, tp2, nf2, ent2 = compute_frame_topo(all_pts)
            frame_feats_all22.append((mp2, tp2, nf2, ent2, len(all_pts)))

        if not frame_feats:
            continue
        arr = np.array([f[:7] for f in frame_feats], dtype=float)
        n_def_avg = np.mean([f[7] for f in frame_feats])
        means = np.nanmean(arr, axis=0)
        def_results.append((gid, pid, *means, n_def_avg))

        arr2 = np.array([f[:4] for f in frame_feats_all22], dtype=float)
        means2 = np.nanmean(arr2, axis=0)
        n_all22_avg = np.mean([f[4] for f in frame_feats_all22])
        all22_results.append((gid, pid, *means2, n_all22_avg))

    def_df = pd.DataFrame(def_results, columns=[
        'gameId', 'playId', 'max_pers', 'tot_pers', 'nfeat', 'entropy',
        'hull_area', 'spread', 'nearest_def_dist', 'n_defenders'])
    all22_df = pd.DataFrame(all22_results, columns=[
        'gameId', 'playId', 'max_pers_22', 'tot_pers_22', 'nfeat_22',
        'entropy_22', 'n_points_22'])
    return def_df, all22_df


if __name__ == "__main__":
    import sys
    week_num = int(sys.argv[1])

    plays = pd.read_csv(f'{DATA_DIR}/plays.csv', usecols=['gameId', 'playId', 'passResult', 'playResult'])
    completed = plays[plays['passResult'] == 'C'].copy()
    completed['outcome'] = np.where(
        completed['playResult'] >= 15, 'breakdown',
        np.where(completed['playResult'] <= 3, 'clean_stop', 'none'))
    completed.loc[completed['outcome'] == 'none', 'outcome'] = np.nan
    eligible = completed.dropna(subset=['outcome'])

    np.random.seed(1000 + week_num)
    game_ids_in_week = set()
    for chunk in pd.read_csv(f'{DATA_DIR}/week{week_num}.csv', usecols=['gameId'], chunksize=2_000_000):
        game_ids_in_week.update(chunk['gameId'].unique())
    week_eligible = eligible[eligible['gameId'].isin(game_ids_in_week)]
    bd = week_eligible[week_eligible.outcome == 'breakdown']
    cs = week_eligible[week_eligible.outcome == 'clean_stop']
    print(f"week{week_num}: available breakdown={len(bd)}, clean_stop={len(cs)}")
    n_bd = min(40, len(bd))
    n_cs = min(40, len(cs))
    sample = pd.concat([
        bd.sample(n=n_bd, random_state=1000 + week_num),
        cs.sample(n=n_cs, random_state=2000 + week_num)])

    def_df, all22_df = process_week(week_num, sample)
    if def_df is not None:
        def_df = def_df.merge(sample[['gameId', 'playId', 'outcome']], on=['gameId', 'playId'])
        all22_df = all22_df.merge(sample[['gameId', 'playId', 'outcome']], on=['gameId', 'playId'])
        def_df.to_csv(f'week{week_num}_def.csv', index=False)
        all22_df.to_csv(f'week{week_num}_all22.csv', index=False)
        print(f"week{week_num}: processed {len(def_df)} plays successfully")
    else:
        print(f"week{week_num}: NO MATCHES FOUND")

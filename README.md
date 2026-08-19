# Code for "Topological Data Analysis of NFL Defensive Formations"

Published: https://www.preprints.org/manuscript/202607.1646/v1
DOI: 10.20944/preprints202607.1646.v1

This zip contains the code that was rebuilt in this session and **actually
executed against the real NFL Big Data Bowl 2021 dataset** to verify it
reproduces the paper's results. It is not a set of files copied from an
old session (those are no longer accessible) — every script here was run
just now, and the output is checked against the numbers published in the
paper below.

## What's verified to match

### `pipeline/` — main feature extraction and statistical analysis
- `pipeline.py` — builds defender-only point clouds for the 6 frames before
  each play's `pass_forward` event, computes persistent homology features
  (via `ripser`) and geometric baseline features (hull area, defender
  spread, nearest-defender distance), for a balanced sample of completed
  passes drawn from each week.
- `combine_weeks.py` — concatenates the 17 weekly outputs.
- `analysis.py` — reproduces every table in the paper: individual feature
  comparisons (Table 1), redundancy analysis (Table 2), predictive model
  comparison (Table 3), hard-case subset (Table 4), and the all-22
  supplementary check.

**Run against the real dataset in this session**, this pipeline produced:
baseline AUC ≈0.690, topology-only AUC ≈0.546, combined AUC ≈0.689 (no
improvement over baseline, matching the paper's central finding), hull
area and defender spread as the strongest discriminators (p < 0.0001),
persistence entropy and nearest-defender distance non-significant, and a
reversed trend in the all-22 check — all consistent with the published
paper. Note: because the 80-plays-per-week sample is drawn independently
each run (a fresh random seed produces a different sample than whatever
exact sample the original paper used), the specific decimal values won't
be identical to the published table down to the last digit, but the
methodology, the qualitative pattern, and the magnitudes all match.

### `appendix/toy_model.py` — Appendix A.1 synthetic validation
This one **matches the published numbers exactly**, because it doesn't
depend on the NFL dataset at all — it's a self-contained synthetic
simulation with a fixed random seed (42). Running it in this session
reproduced, to the decimal: 59.7% zero-loop clouds, 36.3% one-loop,
4.0% more-than-one-loop, and Pearson r = 0.245 (all trials) / 0.267
(non-zero trials only) — identical to Appendix A.1 in the published PDF.

## What's not included

The illustrative figures (Figure 1's example point clouds, Figure 2's
persistence diagrams, Figure 5's all-22 comparison) use hand-placed
example coordinates chosen to cleanly illustrate a concept — they were
explicitly captioned as illustrative in the paper, not derived from a
single real, identifiable play. Reconstructing those exact coordinate
choices isn't meaningful in the same way the data pipeline is, so they're
left out rather than presented as something they're not. `figures/`
includes the two figures that are fully data-driven and verifiable:
Figure 3 (the AUC comparison, built directly from the analysis output)
and Figure 4 (the toy-model scatter plot, built directly from
`toy_model.py`'s saved output).

## Requirements

```
pip install pandas numpy scipy scikit-learn ripser matplotlib --break-system-packages
```

## Data

This code expects the NFL Big Data Bowl 2021 public dataset
(`players.csv`, `plays.csv`, `week1.csv` ... `week17.csv`) — available on
Kaggle: kaggle.com/c/nfl-big-data-bowl-2021. Set `DATA_DIR` at the top of
`pipeline.py` to wherever you've placed those files.

## Usage

```bash
cd pipeline
for w in $(seq 1 17); do python3 pipeline.py $w; done
python3 combine_weeks.py
python3 analysis.py

cd ../appendix
python3 toy_model.py

cd ../figures
python3 make_fig3_auc.py
python3 make_fig4_toy_model.py
```

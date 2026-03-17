## Leaderboard Baseline v3 Submission

This submission uses **PSD + Logistic Regression** with exported weights
for improved accuracy. If weights are not set, it falls back to tuned
energy + spectral flatness baselines.

### How to export weights
Run:
```
python scripts/export_psd_logreg_weights.py
```
This writes `results/psd_logreg_weights.json` with:
- `weights` (length 6)
- `bias`

Copy those values into `main.py`:
```
LOGREG_WEIGHTS = [ ... 6 floats ... ]
LOGREG_BIAS = ...
```

### Files in the zip package
Required:
- `main.py` — defines `evaluate(filename) -> int`
- `user_reqs.txt` — minimal dependency list

Optional:
- `README_submission.md`

### How to build the submission zip
From repo root:
```
cd submissions/leaderboard_baseline_v3
zip -r ../leaderboard_baseline_v3.zip main.py user_reqs.txt README_submission.md
cd -
```

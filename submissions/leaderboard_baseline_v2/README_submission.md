## Leaderboard Baseline v2 Submission

This submission improves on v1 by using **tuned thresholds** derived from the labeled
training set for both the energy detector and spectral flatness detector. The submission
is still lightweight and self-contained.

### What changed vs v1
- **Tuned thresholds** are embedded as constants:
  - Energy threshold
  - Spectral flatness threshold
- **Decision rule:** predict occupied if **either** detector fires.

### Files in the zip package
Required:
- `main.py` — defines `evaluate(filename) -> int`
- `user_reqs.txt` — minimal dependency list

Optional:
- `README_submission.md`

### How to build the submission zip
From repo root:
```
cd submissions/leaderboard_baseline_v2
zip -r ../leaderboard_baseline_v2.zip main.py user_reqs.txt README_submission.md
cd -
```

### How to test locally
```
python scripts/test_leaderboard_submission.py
```

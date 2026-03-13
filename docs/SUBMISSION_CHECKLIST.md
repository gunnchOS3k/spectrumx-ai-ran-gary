## Submission Checklist — Leaderboard Baseline v1

- **Files present**
  - [ ] `submissions/leaderboard_baseline_v1/main.py` exists.
  - [ ] `submissions/leaderboard_baseline_v1/user_reqs.txt` exists.
  - [ ] (Optional, recommended) `submissions/leaderboard_baseline_v1/README_submission.md` exists.

- **evaluate(filename) behavior**
  - [ ] `evaluate(filename)` is defined in `main.py`.
  - [ ] `evaluate(filename)` accepts a path to a `.npy` IQ file.
  - [ ] Supported input formats include:
        - complex array `(N,)`
        - float array `(N, 2)` interpreted as `[I, Q]`
        - 1D float array `(N,)` (I-only, Q=0)
        - int16 interleaved `(N*2,)` interpreted as `[I0, Q0, I1, Q1, ...]`
  - [ ] `evaluate(filename)` always returns a plain Python `int`.
  - [ ] The only valid return values are `0` (unoccupied) or `1` (occupied).

- **Zip contents**
  - [ ] Submission zip includes at least:
        - `main.py`
        - `user_reqs.txt`
  - [ ] Optional but allowed:
        - `README_submission.md`
  - [ ] No real competition dataset files are included in the zip.
  - [ ] No large or unrelated project files are included.

- **Local testing**
  - [ ] Dependencies installed locally:
        - `pip install -r submissions/leaderboard_baseline_v1/user_reqs.txt`
  - [ ] Local test script passes:
        - `python scripts/test_leaderboard_submission.py`
  - [ ] Test output shows:
        - Per-file `PASS` lines for each test `.npy` file.
        - Final `OVERALL RESULT: PASS`.

- **Notes**
  - [ ] This submission is the **first safe baseline**, not the final best model.
  - [ ] The implementation does **not** depend on unfinished components
        (DetectionPipeline, SSL, anomaly models, calibration, AI-RAN controllers).


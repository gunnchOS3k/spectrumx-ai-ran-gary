## Leaderboard Baseline v1 Submission

This folder contains the **first safe baseline submission** for the SpectrumX competition.
It is intentionally simple and conservative, using only classical detectors that are already
implemented and understood in this repository.

### Baseline used

- **Primary detector:** Spectral Flatness Detector  
  - Computes the power spectral density (PSD) using Welch's method.  
  - Computes spectral flatness = geometric_mean(PSD) / arithmetic_mean(PSD).  
  - Interprets **lower** flatness as "more signal-like" (occupied) and **higher** as "more noise-like".
- **Fallback detector:** Energy Detector  
  - Computes the mean power of the IQ samples and compares against a fixed threshold.

The submission logic is:

1. Load the `.npy` IQ file robustly (multiple shapes and dtypes supported).
2. Run the **SpectralFlatnessDetector**.  
3. If it raises an error or cannot operate safely, fall back to the **EnergyDetector**.  
4. Return a plain Python `int` prediction: `0` (unoccupied) or `1` (occupied).

### Expected input format

`evaluate(filename)` expects `filename` to point to a `.npy` file containing IQ data
in one of the following formats:

- **Complex array**: shape `(N,)`, dtype `complex64` or `complex128`
- **Float array with I/Q columns**: shape `(N, 2)` where column 0 is I, column 1 is Q
- **1D float array**: shape `(N,)`; interpreted as I only with Q=0
- **int16 interleaved** (supported): shape `(N*2,)` with `[I0, Q0, I1, Q1, ...]`

If a file does not match any of these formats, the loader will raise a `ValueError`.

### Files in the zip package

For the **competition submission zip**, include at minimum:

- `main.py` — defines `evaluate(filename) -> int` and contains the baseline logic.
- `user_reqs.txt` — minimal dependency list (`numpy`, `scipy`).

It is safe (but not required by the organizers) to also include:

- `README_submission.md` — this file, for human-readable context.

The package **must not** include:

- Any real competition dataset files.  
- Large artifacts or unrelated project code.

### How to test locally

From the repo root:

```bash
# (Optional) activate your environment
# python -m venv .venv && source .venv/bin/activate

pip install -r submissions/leaderboard_baseline_v1/user_reqs.txt

python scripts/test_leaderboard_submission.py
```

The test script will:

- Import `submissions.leaderboard_baseline_v1.main.evaluate`.
- Find or create one or more small synthetic `.npy` IQ samples.
- Call `evaluate(filename)` and assert that the output is an `int` in `{0, 1}`.
- Print a clear **PASS** or **FAIL** message.

### How to build the submission zip

From the repo root, you can either:

1. Use the helper script:

   ```bash
   python submissions/leaderboard_baseline_v1/zip_submission.py
   ```

   This creates `leaderboard_baseline_v1.zip` containing:

   - `main.py`
   - `user_reqs.txt`
   - `README_submission.md`

2. Or run the equivalent command manually:

   ```bash
   cd submissions/leaderboard_baseline_v1
   zip -r ../leaderboard_baseline_v1.zip main.py user_reqs.txt README_submission.md
   cd -
   ```

### Notes

- This is the **first safe baseline submission**, not the final best model.  
- It deliberately **does not depend on unfinished components** (`DetectionPipeline`, SSL, anomaly models, calibration, or AI-RAN controllers).  
- Future submissions can swap in stronger detectors (e.g., PSD+LogReg or SSL-based models) once they are fully implemented and validated.


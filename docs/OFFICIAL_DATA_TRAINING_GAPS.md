## Official Data Training – March 2026 Gaps

### What is already ready

- **Official dataset loader and downloader**
  - `src/edge_ran_gary/data_pipeline/spectrumx_loader.py`:
    - Uses the official `spectrumx` SDK (`Client`) and `SDS_SECRET_TOKEN` from environment or `.env`.
    - Hard-codes the official dataset UUID `458c3f72-8d7e-49cc-9be3-ed0b0cd7e03d`.
    - Implements `SpectrumXDataset.download()` which pulls the dataset from SDS into a local folder.
    - Provides `load_labeled()` and `load_unlabeled()` that read `.npy` files and `groundtruth.csv`.
- **Feature-based training script (local only)**
  - `scripts/train_feature_detector.py`:
    - Loads labeled IQ samples from a directory + CSV.
    - Extracts a rich set of handcrafted features per 1-second IQ window.
    - Trains and compares logistic regression and linear SVM with stratified cross-validation.
    - Computes accuracy, precision, recall, F1, confusion matrix, and selects a low-FP threshold.
    - Exports a compact `.npz` artifact with: model name, weights, bias, feature normalization, and threshold.
- **Leaderboard wrapper**
  - `submissions/leaderboard_baseline_v1/main.py`:
    - Provides `evaluate(filename)` as required by the competition.
    - Robustly loads `.npy` IQ files in multiple formats (complex, `(N,)`; float `(N,2)`; 1D float; int16 interleaved).
    - Contains a shared handcrafted feature extractor for 1-second IQ windows.
    - Returns **only** `0` or `1` and is fail-safe (returns `0` on any unexpected error).

### What is missing

- **Explicit, reproducible official data download script**
  - There is no top-level `scripts/download_official_spxdac.py` that:
    - Wraps the SpectrumX SDK usage.
    - Reads `SDS_SECRET_TOKEN` from env or `.env`.
    - Downloads the official dataset UUID to a standard local path such as `data/competition_dataset/`.
    - Is clearly labeled “official data is local-only and must never be committed.”
- **Labeled subset manifest for supervised training**
  - Official labeled files are discoverable via `SpectrumXDataset`, but:
    - There is no script that builds a **local-only** supervised subset manifest:
      - `data/competition_subset/metadata.csv` with `file,label` columns.
      - `data/competition_subset/iq/` containing only the labeled `.npy` files (copied or symlinked).
    - There is no validation that every row in `metadata.csv` points to an existing file.
- **Shared feature module for training + submission**
  - The feature extractor currently lives inside `submissions/leaderboard_baseline_v1/main.py` and is imported directly there.
  - `scripts/train_feature_detector.py` imports `_extract_features` and `_load_iq_auto` from the submission wrapper instead of a shared module under `src/edge_ran_gary/detection/`.
  - This makes it harder to evolve features in one place and keep training and inference strictly in sync.
- **Leaderboard wrapper not yet wired to learned artifact**
  - `submissions/leaderboard_baseline_v1/main.py` currently:
    - Extracts features and uses a simple spectral-flatness–based rule as the decision logic.
    - Does **not** yet load `feature_detector_v1.npz` and apply a learned linear model.
  - Result: the leaderboard submission is still a heuristic baseline rather than a trained feature-based detector.
- **End-to-end training and validation helpers**
  - There is no single script that:
    - Verifies the official dataset is present locally.
    - Verifies that the labeled subset manifest exists.
    - Invokes the training script with the canonical paths.
    - Confirms that `feature_detector_v1.npz` is produced and reports its metrics.
  - `scripts/test_leaderboard_submission.py` checks that `evaluate(filename)` returns 0/1 on synthetic samples, but:
    - It does not currently introspect the learned artifact (raw scores/threshold) when present.
    - It does not run a full “official training pipeline sanity check.”

### What will be changed in this PR

1. **Official data download script**
   - Add `scripts/download_official_spxdac.py` that:
     - Uses `SpectrumXDataset` / SpectrumX SDK under the hood.
     - Reads `SDS_SECRET_TOKEN` from environment or `.env`.
     - Downloads dataset UUID `458c3f72-8d7e-49cc-9be3-ed0b0cd7e03d` to `data/competition_dataset/` by default.
     - Prints clear success/failure counts and is safe to rerun without overwriting by default.
     - Includes a header note that official competition data is local-only and must never be committed.

2. **Labeled subset builder**
   - Add `scripts/build_official_labeled_subset.py` that:
     - Uses the downloaded official dataset (via `SpectrumXDataset`) to locate the labeled directory and `groundtruth.csv`.
     - Builds `data/competition_subset/metadata.csv` with `file,label` columns.
     - Copies or symlinks only the labeled `.npy` files into `data/competition_subset/iq/` (LOCAL ONLY; gitignored).
     - Validates that every entry in `metadata.csv` has an existing `.npy` file.

3. **Shared feature module**
   - Create `src/edge_ran_gary/detection/feature_baseline.py` that defines:
     - `DEFAULT_SAMPLE_RATE`
     - `_load_iq_auto`
     - `_compute_psd`
     - `_extract_features`
   - Update:
     - `scripts/train_feature_detector.py`
     - `submissions/leaderboard_baseline_v1/main.py`
     to import these shared utilities instead of duplicating logic.

4. **Learned artifact integration in leaderboard wrapper**
   - Extend `submissions/leaderboard_baseline_v1/main.py` so that `evaluate(filename)`:
     1. Loads IQ robustly.
     2. Extracts shared features.
     3. If `feature_detector_v1.npz` exists alongside `main.py`, loads:
        - model name, weights, bias, feature_names, mean, std, threshold.
        - standardizes features, computes linear score, applies threshold to produce 0/1.
     4. On missing/malformed artifact, falls back to:
        - Spectral-flatness heuristic, then energy detector.
     5. Remains fail-safe and returns 0 on any unexpected errors.

5. **Training defaults and validation helpers**
   - Update `scripts/train_feature_detector.py` defaults to:
     - `--iq-dir data/competition_subset/iq`
     - `--metadata data/competition_subset/metadata.csv`
     - `--file-col file`
     - `--label-col label`
   - Add `scripts/validate_official_training_pipeline.py` to:
     - Confirm official dataset and subset manifest exist.
     - Run training with canonical arguments.
     - Print summary metrics and artifact path.
   - Update `scripts/test_leaderboard_submission.py` to:
     - Use the shared feature path.
     - Test evaluate() on synthetic complex64 and `(N,2)` float samples.
     - If `feature_detector_v1.npz` exists, log raw score and threshold for debugging.

6. **Docs and gitignore**
   - Ensure `.gitignore` contains:
     - `data/competition_dataset/`
     - `data/competition_subset/`
     - `*.npy`
     - `.env`
   - Add `docs/OFFICIAL_TRAINING_RUNBOOK.md` that:
     - Documents how to set `SDS_SECRET_TOKEN`.
     - Shows exact commands to:
       - Download official data.
       - Build the labeled subset manifest.
       - Train the feature-based detector.
       - Validate the pipeline.
       - Build the leaderboard zip.
     - Explicitly warns that raw official data must never be committed, uploaded to Streamlit Cloud, or included in the leaderboard zip.


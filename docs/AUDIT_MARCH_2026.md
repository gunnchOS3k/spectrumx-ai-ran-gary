## COMPLETED

- **Baseline detectors (usable today)**
  - `src/edge_ran_gary/detection/baselines.py`:
    - `EnergyDetector`: fully implemented, self-contained, no external artifacts required.
    - `SpectralFlatnessDetector`: fully implemented, uses `scipy.signal.welch`, no fitted state.
  - **Streamlit integration**:
    - `apps/streamlit_app.py` re-implements compatible `energy_detector` and `spectral_flatness_detector` functions and uses them successfully in the dashboard.
- **Digital twin core + micro-twin utilities**
  - `src/edge_ran_gary/digital_twin/__init__.py`: clean export surface for digital twin utilities.
  - `src/edge_ran_gary/digital_twin/generator.py`, `zones.py`, `dataset_builder.py`: end-to-end synthetic dataset generator producing `.npy` IQ windows and `metadata.csv`.
  - `src/edge_ran_gary/digital_twin/contracts.py`, `samples.py`, `generate.py`, `gary_micro_twin.py`, `cli_generate.py`: completed digital-twin contract types, sample generation helpers, CLI, and micro-twin configuration.
  - `configs/gary_micro_twin.yaml`: concrete, working config for micro-twin generation (used by Streamlit).
- **Streamlit app**
  - `apps/streamlit_app.py`:
    - Robust IQ loader supporting multiple `.npy` formats (complex, `(N,2)` float, `int16` interleaved, 1D float).
    - Baseline detectors wired into UI with controls and visualizations (time plots, constellation, PSD, spectrogram).
    - Safety banner preventing competition-data uploads to cloud; demo and micro-twin hooks for safe testing.
- **Tests**
  - `tests/test_streamlit_import.py`: verifies Streamlit app imports without pulling in heavy/SSL dependencies.
  - `tests/test_digital_twin_contract.py`: validates digital twin contracts and sample generation against `DigitalTwinSample` spec.
- **Docs**
  - Digital twin documentation (`docs/DIGITAL_TWIN.md`, `docs/DIGITAL_TWIN_MICRO_TWIN.md`, `docs/DIGITAL_TWIN_OUTPUT_SPEC.md`) describing generator, metadata, and output spec.
  - Streamlit deployment docs (`docs/STREAMLIT_DEPLOY.md`, `docs/STREAMLIT_CLOUD_DEPLOY.md`) aligned with current `streamlit_app.py` entrypoint.
  - Project planning / status docs (`docs/PROJECT_PLAN.md`, `docs/STATUS_SNAPSHOT.md`, `docs/PROGRESS_REPORT.md`, guides/) reflecting current architecture and roles.

## PARTIAL / IN PROGRESS

- **Detection pipeline (Phase 1)**
  - `src/edge_ran_gary/detection/features.py`:
    - Class `FeatureExtractor` exists with clear docstrings and method stubs (`extract_time`, `extract_freq`, `extract_statistical`, `extract_all`).
    - All feature methods currently return empty dicts; no real feature extraction implemented.
  - `src/edge_ran_gary/detection/predict.py`:
    - `DetectionPipeline` class defined with `predict` and `predict_batch` methods.
    - Core logic is mostly TODO hooks:
      - Features: relies on `FeatureExtractor.extract_all`, which is unimplemented.
      - Model loop: `for model in self.models` body is `pass` (no inference / probabilities).
      - Calibration: `Calibrator` integration is sketched but commented out.
      - Ensemble fusion: placeholder only.
      - Batch path calls `self.predict` but unpacks return incorrectly in comments; current implementation returns dict, not `(pred, conf, meta)`.
    - Net effect: `DetectionPipeline` is **not usable** end-to-end without further implementation.
- **SSL / advanced models**
  - `src/edge_ran_gary/detection/ssl.py`:
    - `EncoderSSL` class exists, but:
      - `ssl_method`, encoder/backbone, projector, predictor, augmentations, loss, and training schedule are all TODOs.
      - `pretrain`, `save_checkpoint`, and `load_checkpoint` are stubs (`pass`).
      - `encode` currently just echoes features through (`return features`).
    - No trained SSL artifact or integration into `DetectionPipeline`.
  - `src/edge_ran_gary/detection/anomaly.py`:
    - `AnomalyModel` skeleton with config wiring and method shells.
    - `fit`, `predict`, `predict_proba`, and `score_samples` are mostly placeholders (return zeros).
  - `src/edge_ran_gary/detection/calibrate.py`:
    - `Calibrator` class shell with config and high-level API.
    - `calibrate`, `predict_proba`, `compute_ece`, and `select_threshold` are mostly TODOs or no-ops; `is_fitted` is set to `True` without real fitting.
  - Net effect: SSL + anomaly + calibration stack is **designed but not implemented**.
- **PSD+LogReg baseline**
  - `apps/streamlit_app.py`:
    - `psd_logreg_detector` is explicitly a placeholder returning `None`.
    - UI exposes “PSD+LogReg” option but falls back to an info message and `(0, 0.0)` outputs.
  - `docs/guides/ANANYA_GUIDE.md`:
    - Provides a detailed template for `PSDLogRegDetector` in `src/edge_ran_gary/detection/baselines.py` but this class is **not actually implemented** in `baselines.py`.
  - Net effect: PSD+LogReg is **planned but not realized**; there is no fitted PSD+LogReg artifact in the repo.
- **Digital twin dataset builder CLI**
  - `src/edge_ran_gary/digital_twin/dataset_builder.py`:
    - Core `build_synth_dataset` is implemented and callable.
    - CLI `main()` default `--config` path is `configs/digital_twin_gary.yaml`, which does **not** match the current micro-twin config file (`configs/gary_micro_twin.yaml`).
    - Usable when `--config` is explicitly overridden; default path is stale.

## MISSING FOR LEADERBOARD SUBMISSION

- **Leaderboard submission package layout**
  - No `submissions/` folder exists yet.
  - There is no `submissions/leaderboard_baseline_v1/` directory, and thus no:
    - `main.py` with `evaluate(filename) -> 0 or 1`.
    - `user_reqs.txt` restricted dependency list.
    - `README_submission.md` explaining baseline, input format, and packaging steps.
    - Optional `zip_submission.py` helper.
- **Evaluate entrypoint**
  - No standalone module today exposes a simple `evaluate(filename)` function meeting organizer contract:
    - Accept path to `.npy` IQ file.
    - Robustly load and normalize IQ shapes/dtypes.
    - Run a baseline detector and return a plain Python `int` 0/1.
  - Streamlit app contains robust IQ loading logic and baseline detectors, but this is embedded in UI code and not packaged for the competition’s `main.py` interface.
- **Dependency manifest for submission**
  - Root-level `requirements.txt` and `pyproject.toml` exist for the full repo/tooling stack.
  - There is **no minimal `user_reqs.txt`** scoped to the leaderboard bundle:
    - Needs at least `numpy` and `scipy` (for PSD / Welch) and possibly small extras if we pick a more advanced baseline.
- **Local submission validation**
  - No script exists to exercise a submission `main.evaluate` on dummy `.npy` files:
    - Missing `scripts/test_leaderboard_submission.py` (or similar) that:
      - Imports `submissions/leaderboard_baseline_v1/main.py`.
      - Calls `evaluate(filename)` on sample `.npy` files.
      - Asserts return type is `int` in `{0, 1}`.
- **Submission process documentation**
  - No dedicated `docs/SUBMISSION_CHECKLIST.md` describing:
    - Required files in the zip (`main.py`, `user_reqs.txt`).
    - Constraints on including data (must not bundle real competition datasets).
    - How to run the local test script and expected outputs.

### Baseline choice implications for v1 submission

- **Reliable today**
  - `EnergyDetector` and `SpectralFlatnessDetector` are the only fully implemented, stateless, and well-tested detectors.
  - Streamlit already uses them successfully with a variety of IQ formats.
- **Not ready today**
  - PSD+LogReg detector: design and guide exist, but:
    - No `PSDLogRegDetector` class in `baselines.py`.
    - No serialized model artifact (e.g., `.pkl` or `.joblib`) checked into the repo.
  - DetectionPipeline + SSL + anomaly + calibration stack: interface-only; no usable end-to-end implementation.

**Conclusion for v1 leaderboard package:**  
Use a **simple baseline detector** (SpectralFlatnessDetector where possible, otherwise EnergyDetector) wired directly from `src/edge_ran_gary/detection/baselines.py` with a robust `.npy` loader adapted from `apps/streamlit_app.py`. Avoid `DetectionPipeline`, SSL, anomaly models, and PSD+LogReg for the first leaderboard-ready submission.


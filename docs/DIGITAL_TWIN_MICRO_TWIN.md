# Gary Micro-Twin: Competition-Safe Synthetic IQ Generator

## What it is

The **Gary Micro-Twin** is a small, reproducible scenario generator that outputs:

- **1-second IQ windows** (complex64) with binary label 0 (noise only) or 1 (structured signal).
- **Zone-aware metadata**: `zone_id`, landmark name, seed, `snr_db`, `signal_type`, sample_rate, etc.
- **Reproducibility**: same seed + zone_id + label → identical IQ and metadata.

It uses three anchor zones:

1. **Gary City Hall** — civic center, moderate traffic  
2. **West Side Leadership Academy** — one high school, variable occupancy  
3. **Gary Public Library & Cultural Center** — community hub, steady baseline  

Config: `configs/gary_micro_twin.yaml`.  
Code: `src/edge_ran_gary/digital_twin/` (generator, zones, `gary_micro_twin.py`, `cli_generate.py`).

## What it is not

- **Not a replacement for SpectrumX competition data.** Evaluation and submission must use the official dataset.
- **Not a full propagation model.** No ray tracing or GIS; it’s a lightweight, zone-parameterized signal + noise generator for ML and demos.

## How it feeds ML tests

- **Controlled experiments:** Generate data per zone / SNR / signal type to test detector robustness and fairness.
- **Ablations:** Compare performance across zones or signal types.
- **Reproducibility:** Fixed seeds and configs give repeatable datasets.

## How it supports Streamlit demos

- **“Generate Micro-Twin Demo Data”** in the app runs the generator for a small N (e.g. 9–15), then you pick a file and see IQ / PSD / spectrogram (and prediction if a model is loaded).

## How it supports the AI-RAN portfolio narrative

- **Detector → occupancy probability → digital twin → AI-RAN:** The micro-twin provides a minimal “Gary” scenario (City Hall, one high school, library) for storytelling and portfolio demos without touching competition evaluation.

## Quick usage

```bash
# Generate 50 samples into data/synthetic/gary_micro_twin
python -m edge_ran_gary.digital_twin.cli_generate --config configs/gary_micro_twin.yaml --n 50 --out data/synthetic/gary_micro_twin
```

Outputs (all under `--out`, typically gitignored):

- `*.npy` — IQ files  
- `metadata.csv` — per-file metadata  
- `summary.json` — counts by zone, signal_type, label  

## Reproducibility

Same `seed`, `zone_id`, and `label` always produce the same IQ and metadata. See `scripts/smoke_test_digital_twin.py` (e.g. `test_reproducibility`).

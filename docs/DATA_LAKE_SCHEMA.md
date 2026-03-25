# Aerial Data Lake and OTA evidence path

## Role in the stack

| Layer | Status in this repo |
|-------|---------------------|
| **Core judged submission** | SpectrumX detector on **official** labeled IQ (offline) |
| **Completed extension** | Gary digital twin + scenario engine + RIC-style controller **abstraction** |
| **Simulation backbone** | DeepMIMO / Sionna RT / AODT **summaries** when files exist |
| **OTA evidence target** | **Not active** until you add captures under `data/ota_evidence/` |

## Manifest

Place a top-level manifest at:

`data/ota_evidence/ota_lake_manifest.json`

Minimum fields:

- `ota_lake_manifest_version` (number)
- `captures` (array of objects aligned with `schemas/aerial_data_lake/ota_capture_record.schema.json`)

## Record semantics

Each capture describes **one** RF observation window:

- **RF:** center frequency, sample rate, duration, antenna description
- **Place:** `site_id` tying the capture to Gary anchor semantics
- **Labels:** optional `detector_label_at_capture` vs `ground_truth_label` for calibration
- **Context:** weather, interference tags, optional GPS

## Mapping to detector and twin

1. **Detector:** Offline training scripts read manifest + `.npy` IQ paths; **not** wired into Streamlit scoring.
2. **Digital twin:** `site_id` joins scenario engine anchors for **domain shift** and **replay** narratives.
3. **Calibration:** `calibration_offset_db` supports threshold / SNR calibration studies.

## Python interface

`src/edge_ran_gary/ota_data_interface.py` defines `OTACaptureRecord`, `default_ota_schema_dict()`, and `load_ota_lake_manifest(repo_root)`.

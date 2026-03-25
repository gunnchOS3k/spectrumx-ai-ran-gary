# ARC-OTA / runtime target (architecture note)

## Status: **not active** in this repository

**ARC-OTA** here means a **lab or field** over-the-air experimentation target that would sit **below** the judged SpectrumX detector (offline training) and **beside** the Aerial Data Lake (evidence indexing).

## How it fits

1. **Judged core:** SpectrumX DAC detector trained and scored on **official** IQ offline.
2. **Completed extension:** Gary digital twin + AI-RAN-style **controller abstraction** (Python state, not E2).
3. **Simulation backbone:** DeepMIMO / Sionna / AODT **exports** loaded for provenance and UI.
4. **pyAerial / Aerial CUDA RAN:** Credible **execution stack** for PHY-heavy experiments when NVIDIA tooling is available **externally**.
5. **Data Lake:** `ota_lake_manifest.json` indexes **OTA captures** (IQ + metadata) for calibration, evaluation, and retraining **off Streamlit**.
6. **ARC-OTA target (future):** Define explicitly in your lab docs: radio SKU, frequency plan, sync to `site_id`, and how captures map into `schemas/aerial_data_lake/`. This repo provides **schema and manifest shape**, not hardware drivers.

## Execution surface

- Streamlit: **never** performs OTA capture or ARC closed loop.
- External: SDR / chamber / field logger → files → manifest → offline ML and twin replay.

See `docs/EXTERNAL_RUNTIME_GAPS.md` and `docs/DATA_LAKE_SCHEMA.md`.

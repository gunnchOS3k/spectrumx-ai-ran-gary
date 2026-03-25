# Aerial / 6G civic stack roadmap

**Non-negotiable:** **Core judged submission** (SpectrumX detector on official data) stays separate from the **completed extension** and **next scaling** items below.

## Must add now (repo honesty + hooks)

| Item | Path / artifact |
|------|------------------|
| Provenance vocabulary | `src/edge_ran_gary/simulation_provenance.py` |
| pyAerial bridge (stub + PHY hints) | `src/edge_ran_gary/pyaerial_bridge/` |
| OTA / Data Lake interface | `src/edge_ran_gary/ota_data_interface.py`, `schemas/aerial_data_lake/` |
| Drop zones in hook map | `src/edge_ran_gary/simulation_integration_hooks.py` (`data/pyaerial_bridge`, `data/ota_evidence`) |
| Mobility placeholders | `configs/wireless_scene/mobility_flow_profiles.json`, `src/edge_ran_gary/gary_mobility_profiles.py` |
| AODT hook example | `configs/wireless_scene/aodt_scene_hooks.example.yaml` |
| Streamlit architecture cards | `apps/streamlit_app.py` (judge tour) |

## Must add next (lab / GPU / accounts)

| Item | Path / artifact |
|------|------------------|
| Real DeepMIMO export | `data/deepmimo/scenario_summary.json` + NPZ (see `scripts/export_deepmimo_summary.py`) |
| Real Sionna RT export | `data/sionna_rt/` GeoJSON + JSON (see `scripts/export_sionna_rt_summary.py`) |
| AODT twin manifest | `data/aerial_omniverse/twin_manifest.json` + USD scene |
| pyAerial import in env | NVIDIA-provided wheels / conda; probe via `describe_pyaerial_environment()` |
| OTA capture manifest | `data/ota_evidence/ota_lake_manifest.json` + IQ sidecars |
| ARC-OTA runtime target | Document target SKU / SDR in lab notes (out of repo or `docs/` only) |

## Nice to have later

| Item | Notes |
|------|--------|
| Live O-RAN E2 / O1 traces | xApp/rApp packaging; not claimed today |
| Full city vegetation / clutter meshes | Omniverse pipeline |
| GTFS / OSM mobility fusion | Feeds `data/mobility_traces/` |
| Automated calibration from OTA | Batch jobs reading Data Lake schema |

## Operational RAN target framing

**Aerial CUDA-Accelerated RAN** is the **credible** execution target for PHY-heavy experiments once NVIDIA stacks are available. This repo remains **integration-ready** and **OTA-ready** without implying those systems run in CI or Streamlit Cloud.

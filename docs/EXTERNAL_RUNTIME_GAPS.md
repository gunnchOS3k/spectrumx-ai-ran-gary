# External runtime and infrastructure (honest boundaries)

Streamlit Cloud and this repository are **visualization, provenance, and integration** layers. They **do not** execute full AODT scene generation, pyAerial/cuPHY, live OTA capture, or ARC-OTA lab runtimes.

## What is done **in repo** (no GPU required)

| Item | Location |
|------|-----------|
| Detector packaging and offline evaluation workflow | `submissions/`, judge metrics CSVs |
| Gary digital twin UI + scenario engine (proxies) | `src/edge_ran_gary/gary_scenario_engine.py`, Streamlit |
| Simulation **manifest** loaders (JSON / GeoJSON) | `src/edge_ran_gary/simulation_integration_hooks.py` |
| Data **provenance** + **execution surface** labels | `src/edge_ran_gary/simulation_provenance.py` |
| Analytic DeepMIMO-shaped export script | `scripts/export_deepmimo_summary.py` |
| Analytic Sionna-shaped export script | `scripts/export_sionna_rt_summary.py` |
| **Validation** of on-disk exports (read-only) | `scripts/validate_simulation_exports.py` |
| pyAerial **probe** + PHY hint types | `src/edge_ran_gary/pyaerial_bridge/` |
| OTA schema + manifest loader | `schemas/aerial_data_lake/`, `src/edge_ran_gary/ota_data_interface.py` |
| AODT checklist + drop-zone expectations | `docs/AODT_EXPORT_CHECKLIST.md`, `configs/wireless_scene/aodt_export_expectations.yaml` |

## Ready to run **locally** (developer machine)

| Workflow | Notes |
|----------|--------|
| `streamlit run apps/streamlit_app.py` | Loads bundled `examples/simulation_exports/*` or optional `data/*` |
| `python3 scripts/export_deepmimo_summary.py` | Writes **demo-grade** summaries unless `full_solver` criteria are met honestly |
| `python3 scripts/export_sionna_rt_summary.py` | Analytic path-loss + GeoJSON; not full ray tracing |
| `python3 scripts/validate_simulation_exports.py` | Checks JSON shape under `data/deepmimo` and `data/sionna_rt` |
| `python3 scripts/check_ngc_access.py` | Optional env probe; writes **access_summary.json** (no secrets) |

## Requires **cloud / GPU / lab** (not faked in Streamlit)

| Capability | Requirement |
|------------|-------------|
| **AODT full scene** | NVIDIA Omniverse, AI Aerial tooling, **GPU**, often NGC / program access |
| **Sionna RT ray tracing** | TensorFlow/JAX + Sionna, **GPU** recommended |
| **DeepMIMO full solver** | MATLAB or institutional DeepMIMO pipeline; then honest `full_solver` provenance |
| **pyAerial / cuPHY** | NVIDIA Aerial **container** or bare-metal GPU stack; optional Python wheels |
| **OTA / Data Lake captures** | Lab SDR or field hardware; IQ + metadata **outside** Streamlit |
| **ARC-OTA / runtime target** | Defined in `docs/ARC_OTA_RUNTIME_TARGET.md`; **not** active until lab defines SKU and workflow |

## Streamlit Cloud role

- **Loads** validated JSON/GeoJSON **if present** in the deployed tree (including `examples/simulation_exports/`).
- **Shows** `provenance_tier` (demo summary vs simulation export vs access vs OTA-backed) and `execution_surface_label` (manifest-only vs external regeneration).
- **Never** runs Omniverse, Sionna solvers, pyAerial PHY, or OTA capture.

See also: `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`, `docs/AERIAL_STACK_ROADMAP.md`.

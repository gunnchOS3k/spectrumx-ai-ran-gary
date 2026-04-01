# Provenance and execution legend (research extension)

This legend applies to **simulation / AODT / pyAerial / OTA** hooks only, not the **judged SpectrumX detector**.

## Evidence (six canonical terms)

| User-facing term | Meaning |
|------------------|---------|
| **proxy-only** | Twin/scenario uses analytic proxies; no validated solver export on disk for that layer (typical DeepMIMO/Sionna when not loaded). |
| **loaded demo** | Parsed manifest from `examples/simulation_exports/` or export with downgraded `export_provenance` (not `full_solver`). |
| **loaded simulation export** | Parsed artifact under `data/` treated as operator or `full_solver`-backed. |
| **installer-ready** | Aerial `access_summary.json` present (env/installer probe); **not** a scene solve. |
| **external-runtime-required** | Real PHY/twin/OTA IQ for that integration requires **GPU / lab / cloud**; Streamlit does not execute it (e.g. empty pyAerial path). |
| **OTA-backed** | Valid `ota_lake_manifest.json` loaded; IQ files still **outside** Streamlit. |

Machine key: `provenance_tier` (e.g. `demo_summary_loaded`). User-facing: `canonical_evidence_status` / `provenance_label` on hook results (`simulation_provenance.py`).

## Execution surface (three terms)

| Term | Meaning |
|------|---------|
| **manifest-load-only** | This app loads JSON/GeoJSON for display only. |
| **external-runtime-required** | Regenerating or running full solvers needs external MATLAB, Sionna GPU, Omniverse, Aerial, etc. |
| **lab-OTA-workflow** | OTA capture and labeling are lab/field; manifests may be read in-app. |

Field: `canonical_execution_status`. Long text: `execution_surface_label`.

## Rule

**Streamlit** = visualization + provenance. **Never** full AODT generation, Sionna ray trace, pyAerial PHY, or OTA capture.

See `docs/EXTERNAL_RUNTIME_GAPS.md` and `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`.

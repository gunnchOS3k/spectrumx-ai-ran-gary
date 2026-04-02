# What runs where: external runtime boundaries

This document is the **research boundary** for the Gary **research extension** (simulation, AODT, pyAerial, OTA). It does **not** redefine the **judged SpectrumX** submission.

## Evidence vs execution (quick map)

| Canonical evidence term | Typical execution surface |
|-------------------------|---------------------------|
| proxy-only | manifest-load-only |
| loaded demo | manifest-load-only |
| loaded simulation export | manifest-load-only (display) + **external-runtime-required** to regenerate |
| installer-ready | manifest-load-only |
| external-runtime-required | external-runtime-required |
| OTA-backed | lab-OTA-workflow |

Full definitions: `docs/PROVENANCE_LEGEND.md`.

---

## In this repository (no GPU required)

| Capability | Location / notes |
|------------|------------------|
| Judged detector packaging | `submissions/`, offline `evaluate()` |
| Gary twin UI + scenario **proxies** | `gary_scenario_engine.py`, Streamlit |
| Manifest **loaders** (JSON / GeoJSON) | `simulation_integration_hooks.py` |
| **Canonical** evidence + execution labels | `simulation_provenance.py` |
| Analytic DeepMIMO-shaped export | `scripts/export_deepmimo_summary.py` |
| Analytic Sionna-shaped export | `scripts/export_sionna_rt_summary.py` |
| Export **validation** (read-only) | `scripts/validate_simulation_exports.py` |
| pyAerial **import probe** + PHY/MAC **typed abstractions** | `pyaerial_bridge/phy_interface.py` |
| OTA schema + manifest loader | `ota_data_interface.py`, `schemas/aerial_data_lake/` |
| AODT checklist + example manifests | `docs/AODT_EXPORT_CHECKLIST.md`, `examples/simulation_exports/aerial_omniverse/` |

**Architectural targets only (not active execution):** ARC-OTA lab framing — `docs/ARC_OTA_RUNTIME_TARGET.md`.

---

## Ready on a **local developer machine**

| Workflow | Requirement |
|----------|-------------|
| `streamlit run apps/streamlit_app.py` | Python deps; `PYTHONPATH` includes repo root |
| Root wrapper for Cloud | `streamlit_app.py` imports `apps.streamlit_app` |
| Export scripts | No GPU; outputs may be **demo-grade** unless provenance says `full_solver` |
| `python3 scripts/check_ngc_access.py` | Optional env vars; writes **presence-only** `access_summary.json` |

---

## Requires **GPU / cloud / institutional** infrastructure

| Capability | Why external |
|------------|----------------|
| **AODT full scene** | NVIDIA Omniverse + AI Aerial tooling, GPU, often NGC / program access |
| **Sionna RT ray tracing** | TensorFlow/JAX + Sionna; GPU strongly recommended |
| **DeepMIMO full solver** | MATLAB or institutional DeepMIMO pipeline; then honest `full_solver` in `export_provenance` |
| **pyAerial / cuPHY** | NVIDIA Aerial **container** or bare-metal GPU; optional licensed wheels |
| **Aerial CUDA-Accelerated RAN** | Operational PHY/MAC target **outside** this repo |

---

## Requires **lab / field** infrastructure

| Capability | Notes |
|------------|--------|
| **OTA / Data Lake captures** | SDR or field logger; IQ + metadata; `ota_lake_manifest.json` indexes files **not** recorded by Streamlit |
| **ARC-OTA runtime target** | Radio SKU, sync, labeling pipeline — **not** implemented as live code in this repo |

---

## Streamlit Cloud (and this app)

- **Does:** load committed `examples/simulation_exports/*` and optional `data/*` if present; show **evidence** and **execution** labels.
- **Does not:** run Omniverse, Sionna solvers, pyAerial PHY, OTA capture, or ARC-OTA loops.

---

## Related documentation

- `docs/uml/README.md` — UML index: current vs future deployment, manifest ingestion, provenance states  
- `docs/PROVENANCE_LEGEND.md` — six evidence terms + three execution terms  
- `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md` — export scripts, env var **names**  
- `docs/AODT_EXPORT_CHECKLIST.md` — full AODT vs manifest-only  
- `docs/PYAERIAL_BRIDGE.md` — bridge scope and notebook  
- `docs/DATA_LAKE_SCHEMA.md` — OTA record semantics  
- `docs/AERIAL_STACK_ROADMAP.md` — roadmap with repo paths  

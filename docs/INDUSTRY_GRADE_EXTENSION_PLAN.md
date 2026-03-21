# Industry-grade Gary extension — capabilities, proxies, and next scaling

This document summarizes the **Completed Research Extension** (Judge tab: *Completed Research Extension*) after the **industry-grade / postdoctoral demo** pass. It keeps three buckets explicit:

1. **Core judged submission** — SpectrumX DAC detector on **official competition data** (training/eval offline; **not** loaded in Streamlit).
2. **Completed extension** — Site-aware digital twin + proxy propagation view + AI-RAN-style controller demo in `apps/streamlit_app.py` (`_render_judge_gary_micro_twin_3d`).
3. **Next realism scaling** — DeepMIMO channels, Sionna RT ray-tracing; **hooks and directories only** until parsers and outputs exist.

## Current capabilities (simulation-backed vs storytelling)

| Capability | Type |
|------------|------|
| Extruded anchor footprints + scenario-driven coexistence tint | **Implemented** (approximate GIS; not a surveyed BIM) |
| Hypothetical gNB positions, demand disks, interference disks + **polygon interference proxies** | **Proxy / storytelling** |
| **PathLayer** gNB→centroid **serving-link segments** + tooltip LOS-fraction **proxy** | **Proxy** (straight line; **not** Sionna RT) |
| **Per-site table**: coverage score, propagation challenge, LOS/NLOS label, penetration dB equiv., blockage | **Deterministic proxy** from scenario + height |
| **State → action → KPI** controller narrative tied to live `evaluate()` on **synthetic** demo IQ | **Implemented demo logic**; KPIs remain proxies |
| DeepMIMO / Sionna **file drops** under `data/simulation/deepmimo/`, `data/simulation/sionna_rt/` | **Directories + stub loaders** (`simulation_integration_hooks.py`) |
| Optional assets: `submissions/submission_metrics.csv`, `docs/final_report_figures.yaml`, `configs/gary_micro_twin.yaml` | **Optional**; UI lists presence only |

## What is proxy (must stay labeled in UI)

- All map overlays except “there is a building footprint here” are **hypothetical** or **aggregated fiction** for coexistence storytelling.
- Propagation metrics are **current proxy propagation view** — not ray-traced, not DeepMIMO CIRs, not drive tests.
- Controller actions are a **discrete policy stub** for demonstration, not a deployed RIC.

## What is next to integrate

- **Sionna RT:** `coverage_grid.geojson`, `path_loss_summary.json` → pydeck layer + panel replacements for proxy table.
- **DeepMIMO:** `channel_features.npz`, `scenario_meta.json` → feature / SINR columns and optional map overlay.
- **Judged detector:** unchanged; extension consumes only **public** demo IQ in-app.

## Code entry points

- UI: `apps/streamlit_app.py` → `_render_judge_gary_micro_twin_3d`
- Hooks: `src/edge_ran_gary/simulation_integration_hooks.py`

## Run locally

```bash
streamlit run apps/streamlit_app.py
```

**Accounts:** None required for the extension. **Dependencies:** `requirements.txt` plus **`pydeck`** for the 3D tab; **`pandas`** improves the propagation table (fallback captions if absent).

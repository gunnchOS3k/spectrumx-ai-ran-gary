# Simulation backbone — hooks, proxies, and scaling path

This document describes how the **Completed Research Extension** connects to **simulation-backed realism** without blurring the **core judged** SpectrumX detector.

## Current proxy layers (always labeled in UI)

| Layer | Role | Truth |
|-------|------|--------|
| Extruded footprints | Anchor geometry | Approximate GIS / storytelling |
| Green **coverage halos** | Heat-style abstraction | **Proxy** — size ∝ coverage score, not SINR map |
| Cyan links | gNB → site | Straight-line **proxy**, not ray tracing |
| Violet disks | User demand | Narrative **proxy** |
| Orange markers | Propagation stress | **Proxy** from scenario + height |
| Red IF disks / polygons | Coexistence | **Proxy**, not identified emitters |

## Integration-ready drop zones (repo root)

| Path | Purpose | Legacy alternate |
|------|---------|------------------|
| `data/deepmimo/` | DeepMIMO-style CSI / scenario / NPZ / JSON / CSV | `data/simulation/deepmimo/` |
| `data/sionna_rt/` | Sionna RT GeoJSON coverage, path-loss JSON | `data/simulation/sionna_rt/` |
| `data/aerial_omniverse/` | NVIDIA **AI Aerial / Omniverse** exports (e.g. USD refs, twin manifest, RF overlay meta) | — |
| `configs/wireless_scene/` | Layer visibility, anchor overrides | — |
| `configs/ric/` | Example **Near-RT RIC / xApp** YAML stubs (non-operational) | — |

The Streamlit app **scans** these paths (via `simulation_integration_hooks.py`) and **fails gracefully** if empty. **Stub loaders** return no overlay until real parsers exist.

## Pillar 1 — DeepMIMO

- **Contributes:** Reproducible **site-specific MIMO channels**, CSI features, scenario matrices.
- **Connects:** Propagation / SINR panels; optional map overlays; ML / controller replay.
- **Status in repo:** **Integration-ready** — drop files; wire `load_deepmimo_overlay_stub` when ready.
- **Accounts / compute:** Typically **local or HPC** MATLAB/Python pipeline; no cloud account required for the **hook** itself.

## Pillar 2 — Sionna RT

- **Contributes:** **Ray-traced** path loss, materials, diffraction; LOS maps.
- **Connects:** Replace link + halo **proxies** with GeoJSON / grid layers in pydeck.
- **Status:** **Next scaling path** — same as above; stub until adapter.
- **Compute:** GPU helpful for large scenes; optional **NVIDIA** ecosystem alignment.

## Pillar 3 — NVIDIA AI Aerial / Omniverse (AODT)

- **Contributes:** **Large-scale digital twin** + RF visualization workflows when run **outside** this repo.
- **Connects:** Export metadata or lightweight scene descriptors into `data/aerial_omniverse/` for future UI loaders.
- **Status:** **Next scaling path** — **not bundled**; typically requires **NVIDIA accounts**, **Omniverse**, and **GPU** for full fidelity.
- **Truthfulness:** The **judged** detector does **not** use Aerial unless you explicitly integrate it in code; the app only documents **hooks**.

## O-RAN / RIC config stubs

`configs/ric/` is for **documentation and future** policy files (e.g. example xApp thresholds). It does **not** connect to a live Near-RT RIC in the current Streamlit build.

## Code entry points

- `src/edge_ran_gary/simulation_integration_hooks.py` — `describe_all_integration_drop_zones`, stubs  
- `apps/streamlit_app.py` — `_render_judge_gary_micro_twin_3d`  

## Run locally

```bash
streamlit run apps/streamlit_app.py
```

**No new accounts** are required to run the app; **Aerial / Omniverse** are **optional future** dependencies for the scaling path only.

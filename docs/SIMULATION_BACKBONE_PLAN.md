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

The Streamlit app **scans** these paths (via `simulation_integration_hooks.py`) and **fails gracefully** if empty. **Summary loaders** validate JSON/GeoJSON when present; **stub overlay** helpers remain for future deck wiring beyond summaries.

## Repo-bundled demo summaries + load priority

| Location | UI label when parser succeeds |
|----------|-------------------------------|
| `data/deepmimo/`, `data/sionna_rt/`, `data/aerial_omniverse/` (and legacy `data/simulation/*`) | **Loaded (simulation export)** |
| `examples/simulation_exports/deepmimo/`, `.../sionna_rt/`, `.../aerial_omniverse/` | **Loaded (demo summary)** |

**Default loader behavior** (`demo_only=False`): try **simulation** directories first; if nothing **valid** is found, try **demo** directories. **Real exports always win** when they validate.

**Demo-only mode** (`demo_only=True`, Streamlit button *Use bundled demo summaries*): scan **only** `examples/simulation_exports/*` — ignores `data/` (for consistent cloud demos).

**Streamlit Cloud:** runtime-created files under `data/` usually **do not persist** across app restarts. Committed files under `examples/simulation_exports/` **do** persist with the deployment.

Loader return dict keys (among others): `loaded`, `source_kind` (`simulation` \| `demo` \| `absent`), `status_label` (`Loaded (simulation export)` \| `Loaded (demo summary)` \| `Not loaded`), `load_mode`, `path` / `summary_json_path`.

## Pillar 1 — DeepMIMO

- **Contributes:** Reproducible **site-specific MIMO channels**, CSI features, scenario matrices.
- **Connects:** Propagation / SINR panels; optional map overlays; ML / controller replay.
- **Status in repo:** **JSON + optional NPZ metadata** — drop `scenario_summary.json` / `scenario_meta.json`; see schema section below.
- **Accounts / compute:** Typically **local or HPC** MATLAB/Python pipeline; no cloud account required for the **hook** itself.

## Pillar 2 — Sionna RT

- **Contributes:** **Ray-traced** path loss, materials, diffraction; LOS maps.
- **Connects:** Replace link + halo **proxies** with GeoJSON / grid layers in pydeck.
- **Status:** **JSON + optional GeoJSON** — validated summaries + optional **`layer-sionna-coverage-geojson`** map overlay.
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

---

## Exact filenames & parser rules (`simulation_integration_hooks.py`)

The UI shows **Loaded** only when the in-repo parser **accepts** the file (JSON decode + field validation, or GeoJSON geometry check). Empty JSON or wrong shape → **not loaded** + `parser_note`.

### DeepMIMO — `data/deepmimo/` (legacy: `data/simulation/deepmimo/`) **then** `examples/simulation_exports/deepmimo/`

| File | Role |
|------|------|
| `scenario_summary.json` | Preferred summary (first match wins with `scenario_meta.json`) |
| `scenario_meta.json` | Alternate summary name |
| `channel_features.npz` | Optional sidecar — **metadata only** (array names/shapes), never marked “loaded” alone |

**Minimum accepted signal (any one):** `scenario_name`, or `num_bs` / `num_users`, or `los_links` / `nlos_links`, or `deepmimo_export_version` / `deepmimo_version`, or non-empty `channel_statistics`, or non-empty `sites` (under `deepmimo` wrapper or root). Nested `deepmimo: { ... }` is supported.

**Example minimal JSON:**

```json
{
  "deepmimo_export_version": "1.0",
  "deepmimo": {
    "scenario_name": "my_scene",
    "num_bs": 4,
    "num_users": 200
  }
}
```

### Sionna RT — `data/sionna_rt/` (legacy: `data/simulation/sionna_rt/`) **then** `examples/simulation_exports/sionna_rt/`

| File | Role |
|------|------|
| `propagation_summary.json` | Primary JSON (first match with `path_loss_summary.json`) |
| `path_loss_summary.json` | Alternate JSON name |
| `coverage_grid.geojson` | Optional coverage grid (also `coverage.geojson`, `sionna_coverage.geojson`) |

**Loaded if:** JSON validates **or** GeoJSON is a `FeatureCollection` / `Feature` with geometry. GeoJSON-only loads show `parser_note` suggesting adding JSON for richer metrics.

**Map:** Valid GeoJSON is passed to pydeck as **`GeoJsonLayer`** id **`layer-sionna-coverage-geojson`**. Per-site proxy **tables/charts** stay scenario-driven unless you extend mapping.

**Example JSON keys:** `scenario_name`, `mean_path_loss_db`, `los_fraction`, `frequency_ghz`, nested `sionna: { ... }`, or `cells` list, or `sionna_export_version` / `solver`.

### NVIDIA AI Aerial / Omniverse — `data/aerial_omniverse/` **then** `examples/simulation_exports/aerial_omniverse/`

| File | Role |
|------|------|
| `overlay_summary.json` | Manifest-style JSON |
| `twin_manifest.json` | Alternate name |

**Minimum accepted signal:** `scene_name` or `usd_path`, or `aerial_export_version` / `omniverse_kit_version`, or non-empty `assets` list.

**Honesty:** This repo does **not** ship Omniverse Kit, Aerial sim binaries, or USD scene geometry. **Loaded** means **only** that a small JSON manifest on disk validated — **not** that the twin is interactively running. Full workflows typically need **external program access**, **GPU**, and often a **NVIDIA account** / **6G Developer Program** enrollment.

### Bundled demo files (validated names)

Shipped under `examples/simulation_exports/<pillar>/` with final filenames (`scenario_summary.json`, `propagation_summary.json`, `coverage_grid.geojson`, `twin_manifest.json`). They load automatically as **fallback** when `data/*` has no valid summary, or as **only** source when Streamlit **demo-only** mode is on.

---

## UI surfaces

- **Simulation data sources** — buttons: bundled demo-only vs data-first; Cloud persistence note.
- **Simulation summaries** — compact cards (`status_label`, source type, file path, load mode); expander for tables/JSON.
- **Simulation backbone** — reuses the same load results (no second disk read).
- **Propagation / coverage** — success callout when Sionna/DeepMIMO validated; still labels table/chart proxies honestly.
- **Closed-loop controller** — caption distinguishes **simulation-backed metadata** vs **proxy KPI** inputs.

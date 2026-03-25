# AODT (Aerial Omniverse Digital Twin) — export checklist

## Two distinct phases (do not blur)

### 1) Full scene generation (**external runtime**)

Runs **outside** this repo in NVIDIA **Omniverse** / **AI Aerial** (GPU, often NGC or installer-based access).

Produces:

- USD or USDZ scene assets, materials, optional vegetation and terrain meshes
- Mobility / animation clips or exported trajectories (optional)
- RF overlay or twin metadata consistent with your Aerial workflow

**Not** executed in Streamlit Cloud.

### 2) **Export manifests** for this app (**in repo / drop zone**)

After external generation, copy **summary** artifacts the loaders understand:

| Artifact | Typical path | Purpose |
|----------|----------------|--------|
| Overlay / twin manifest | `data/aerial_omniverse/overlay_summary.json` or `twin_manifest.json` | Parsed by `load_aerial_overlay_summary` |
| Access probe (optional) | `data/aerial_omniverse/access_summary.json` | From `scripts/check_ngc_access.py` — **access confirmed / installer-ready**, not a twin |
| Scene hooks (optional) | `configs/wireless_scene/aodt_scene_hooks.example.yaml` | Copy to a non-example name when you wire real paths |

See `configs/wireless_scene/aodt_export_expectations.yaml` for **expected keys** and placeholders for terrain, vegetation, mobility, and scatterer summaries (schemas for **documentation**, not live solvers).

## Honest status labels

- **Loaded (demo summary)** — bundled `examples/simulation_exports/aerial_omniverse/` stub.
- **Loaded (simulation export)** — valid `data/aerial_omniverse/*` with operator `full_solver` provenance when applicable.
- **Access confirmed / installer-ready** — `access_summary.json` only.
- **Execution surface** — always **external** for full AODT generation; Streamlit reads manifests only (`execution_surface_label` on hook results).

## Related scripts

- `scripts/check_ngc_access.py` — environment presence probe (no secrets in JSON).
- `scripts/bootstrap_aodt_access.py` — optional Docker/bootstrap flags (see runbook).

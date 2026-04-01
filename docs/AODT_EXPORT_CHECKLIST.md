# AODT (Aerial Omniverse Digital Twin) — export checklist

## Two phases (do not blur)

### Phase A — Full scene generation (**external runtime only**)

Runs in **NVIDIA Omniverse / AI Aerial** with **GPU** and often NGC or installer access.

| Step | Check |
|------|--------|
| Scene graph / USD | City or campus scope matches Gary anchor intent |
| **Terrain** | DEM or mesh from GIS / photogrammetry; document CRS |
| **Vegetation** | Instancing or layers; note season / LOD |
| **Materials** | Building façade / ground RF-relevant where possible |
| **Georeferencing** | CRS, origin, tie to lon/lat used in twin |
| **Mobility** | Optional animation, CSV, or GTFS-derived traces |
| **Dynamic scatterers** | Vehicles / crowds as separate export or summary |
| RF overlay | Per your Aerial / AODT workflow (not asserted here) |

**Not** executed in Streamlit or CI in this repo.

### Phase B — **Manifests / summaries** for this app (**in repo**)

After Phase A, place JSON the loader accepts under `data/aerial_omniverse/` (or use bundled **demo** under `examples/simulation_exports/aerial_omniverse/`).

| Artifact | Path | Loader |
|----------|------|--------|
| Twin / overlay summary | `overlay_summary.json` or `twin_manifest.json` | `load_aerial_overlay_summary` |
| Access probe (optional) | `access_summary.json` | `scripts/check_ngc_access.py` → evidence **`installer-ready`**, not a scene |
| Extended fields (optional) | Same JSON | `site_identifiers`, `terrain_summary`, `vegetation_summary`, `georeferencing`, `mobility_summary`, `dynamic_scatterer_summary`, `export_provenance`, `generation_environment` (see `_normalize_aerial_dict`) |

**Example extended manifest:** `examples/simulation_exports/aerial_omniverse/overlay_summary.example.json`  
**Expectations YAML:** `configs/wireless_scene/aodt_export_expectations.yaml`

## Validation expectations (honest)

- Loader marks **loaded** only if JSON parses and has **scene_name** or **usd_path** (or version/assets heuristics).
- Invalid JSON on disk → **not loaded** + parser note (no fake twin).
- **`export_provenance.simulation_grade`:** `full_solver` vs analytic/synthetic drives **loaded demo** vs **loaded simulation export** (via shared provenance rules in hooks).

## Streamlit states (AODT row)

| What you see | Meaning |
|--------------|---------|
| **loaded demo** | Parsed manifest from `examples/…` or downgraded provenance |
| **loaded simulation export** | Parsed `data/aerial_omniverse/*` with simulation-tier provenance |
| **installer-ready** | Only `access_summary.json` (no twin manifest) |
| **proxy-only** | No manifest; twin maps use scenario proxies |
| **external-runtime-required** | Integration target needs Omniverse/GPU to produce real scenes |

Execution surface for AODT is always **external-runtime-required** for regeneration; Streamlit remains **manifest-load-only**.

## Related scripts

- `scripts/check_ngc_access.py` — access probe JSON (no secrets).
- `scripts/bootstrap_aodt_access.py` — optional Docker/bootstrap (see runbook).

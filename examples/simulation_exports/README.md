# Simulation export examples (repo-bundled)

## Bundled demo paths (loaded automatically as **fallback**)

If `data/deepmimo/`, `data/sionna_rt/`, and `data/aerial_omniverse/` have **no** valid summaries, the app can still load **validated** JSON/GeoJSON from:

| Pillar | Path |
|--------|------|
| DeepMIMO | `examples/simulation_exports/deepmimo/scenario_summary.json` |
| Sionna RT | `examples/simulation_exports/sionna_rt/propagation_summary.json` and/or `coverage_grid.geojson` |
| Aerial / Omniverse | `examples/simulation_exports/aerial_omniverse/twin_manifest.json` |

The Streamlit UI labels these as **Loaded (demo summary)**. Files under `data/*` validate as **Loaded (simulation export)** when they parse successfully (priority over examples unless **Demo-only** mode is selected in the app).

## Real exports

Place your own outputs under:

- `data/deepmimo/` — `scenario_summary.json` or `scenario_meta.json`
- `data/sionna_rt/` — `propagation_summary.json`, `path_loss_summary.json`, `coverage_grid.geojson`, …
- `data/aerial_omniverse/` — `overlay_summary.json` or `twin_manifest.json`

## Streamlit Cloud

Runtime writes under `data/` are **not persistent** across app restarts. Repo files (including this folder) **are** available after each deploy—use **bundled demo** or ship summaries in the repo.

## Truthfulness

**Loaded** only after JSON/GeoJSON **parses and validates** in `simulation_integration_hooks.py`. No fake loaded state.

**Aerial / Omniverse:** Full fidelity requires external NVIDIA tooling, GPU, and often NVIDIA account / 6G Developer Program access—not bundled here.

# Simulation export examples (copy into `data/` to test loaders)

These files are **not** loaded automatically from `examples/`. Copy or merge into the drop zones below, then restart Streamlit.

| Target directory | Example file | Purpose |
|------------------|--------------|---------|
| `data/deepmimo/` | `scenario_summary.example.json` | DeepMIMO-style summary — rename to `scenario_summary.json` |
| `data/sionna_rt/` | `propagation_summary.example.json` | Sionna metrics — rename to `propagation_summary.json` |
| `data/sionna_rt/` | `coverage_grid_minimal.example.geojson` | GeoJSON-only load test — rename to `coverage_grid.geojson` |
| `data/aerial_omniverse/` | `twin_manifest.example.json` | Aerial/Omniverse manifest — rename to `twin_manifest.json` |

**Truthfulness:** The app sets **Loaded** only after JSON/GeoJSON **parses and validates** per `simulation_integration_hooks.py`.

**Aerial / Omniverse:** Full fidelity requires **external** NVIDIA tooling, **GPU**, and often a **NVIDIA account** / **6G Developer Program** — not bundled here.

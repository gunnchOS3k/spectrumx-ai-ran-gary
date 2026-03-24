# Applied simulation bridge — runbook

**Scope:** Completed **Gary digital-twin extension** + **simulation backbone** (DeepMIMO, Sionna RT, NVIDIA Aerial / AODT hooks).  
**Not in scope:** Core **judged** SpectrumX DAC detector training, leaderboard packaging, or competition IQ in Streamlit.

## Security (non-negotiable)

- **Never** commit API keys, tokens, or NGC secrets to the repo.
- **Never** hardcode keys in scripts, docs, or logs.
- Scripts read credentials **only** from environment variables. If a variable is missing, behavior is **graceful** (skipped probe, stderr hint) — not a crash that leaks secrets.

### Environment variable names (values belong in your shell / CI / Streamlit secrets)

| Variable | Purpose |
|----------|---------|
| `NGC_API_KEY_AODT` | Optional NGC API key for AODT-related registry checks |
| `NGC_API_KEY_AERIAL_RAN` | Optional NGC API key for Aerial RAN–related checks |
| `ENABLE_AODT_BOOTSTRAP` | Set to `1` to run extended bootstrap steps in `bootstrap_aodt_access.py` |
| `ENABLE_DOCKER_PULLS` | Set to `1` to allow a **best-effort** `docker pull` smoke test (requires Docker + network) |
| `DEEPMIMO_FULL_SOLVER_OK` | Set to `1` **only** after you have produced a **real** DeepMIMO scenario off-box and want `export_deepmimo_summary.py --mode full_solver` to emit `full_solver` provenance |
| `AWS_*`, `AZURE_*` | Optional; recorded as **presence-only** booleans in `access_summary.json` (no values) |

## Status semantics (Streamlit)

| Label | Meaning |
|-------|---------|
| **Not loaded** | No valid manifest / summary / GeoJSON for that pillar. |
| **Loaded (demo summary)** | Valid parse from `examples/simulation_exports/…` **or** `data/…` with `export_provenance.simulation_grade` in `analytic_fallback` / synthetic (export scripts). |
| **Loaded (simulation export)** | Valid `data/…` export **without** analytic downgrade (operator file or `full_solver` provenance). |
| **Access confirmed / installer-ready** | *(Aerial only)* `data/aerial_omniverse/access_summary.json` from `check_ngc_access.py` — **not** a twin or simulation export. |

## One-time environment probe (Aerial / NGC)

From repo root:

```bash
python3 scripts/check_ngc_access.py
```

Writes `data/aerial_omniverse/access_summary.json` with **boolean** key presence and probe outcomes — **never** the key strings.

Optional bootstrap (Docker / flags):

```bash
export ENABLE_AODT_BOOTSTRAP=1   # optional
export ENABLE_DOCKER_PULLS=1    # optional; needs Docker daemon + network
python3 scripts/bootstrap_aodt_access.py
```

Also writes `data/aerial_omniverse/bootstrap_report.json` when bootstrap mode is on.

## Export Sionna-shaped summaries (local)

```bash
python3 scripts/export_sionna_rt_summary.py
# Optional: fail CI if Sionna not importable (does not upgrade simulation_grade)
python3 scripts/export_sionna_rt_summary.py --require-sionna-import
```

**Outputs:** `data/sionna_rt/propagation_summary.json`, `path_loss_summary.json`, `coverage_grid.geojson`  
**UI:** Expect **Loaded (demo summary)** (analytic fallback). For **Loaded (simulation export)**, run a **real** Sionna RT scene and write summaries with `export_provenance.simulation_grade: "full_solver"` (and a solver-backed `engine`).

## Export DeepMIMO-shaped summaries (local)

```bash
python3 scripts/export_deepmimo_summary.py
```

**Outputs:** `data/deepmimo/scenario_summary.json`, `scenario_meta.json`  
**UI:** Default = **Loaded (demo summary)** (analytic).  
For **full_solver** (only when honest):

```bash
export DEEPMIMO_FULL_SOLVER_OK=1   # after real DeepMIMO run elsewhere
python3 scripts/export_deepmimo_summary.py --mode full_solver
```

## Streamlit Cloud

- Repo files under `examples/simulation_exports/` persist after deploy → **Loaded (demo summary)** fallback without writing to `data/`.
- Runtime-only files under `data/` may **not** persist across restarts unless you bake them into the image or use mounted storage.

## Run the app locally

```bash
streamlit run apps/streamlit_app.py
```

## Prerequisites (honest)

- **Sionna RT full ray tracing:** TensorFlow or JAX stack, GPU recommended, Sionna version compatible with your code — **not** fully automated in this repo.
- **DeepMIMO:** Often MATLAB-centric; Python modules vary. This repo’s script is a **bridge** until your institutional pipeline is wired.
- **AODT / Omniverse:** Requires NVIDIA tooling, GPU, program access, and NGC — **access_summary** only confirms env/Docker/key **presence** and coarse registry reachability, not an installed twin.

## Related docs

- `docs/SIMULATION_BACKBONE_PLAN.md` — paths, parsers, provenance rules  
- `docs/MICROTWIN_REALISM_PLAN.md` — UI behavior  

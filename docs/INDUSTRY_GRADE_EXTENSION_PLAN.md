# Industry-grade Gary extension — capabilities, proxies, and next scaling

Three buckets:

1. **Core judged submission** — SpectrumX DAC detector evaluated **offline** on competition-style data; **raw competition IQ is not** shipped inside Streamlit.
2. **Completed extension** — Gary **digital-twin wireless scene** + **`gary_scenario_engine`** (population/devices/pressures) + **closed-loop AI-RAN scene controller** (**RIC-style**, **Near-RT / Non-RT–inspired**) in `_render_judge_gary_micro_twin_3d`.
3. **Next realism scaling** — DeepMIMO, Sionna RT, **NVIDIA AI Aerial / Omniverse** — optional **JSON summary manifests** under drop dirs; UI shows **loaded** vs **not loaded** honestly.

## Scenario engine + controller (summary)

| Piece | Role |
|-------|------|
| `ScenarioInputs` | Preset, calendar context, RF sliders, per-site overrides (staff, attendance, visitors, library mode). |
| `SiteScenarioState` | People, IP/control devices (active), traffic demand, coexistence/coverage pressure, fairness score, sourcing/assumption notes. |
| `select_closed_loop_action` | Discrete policy from **scenario state** + detector belief + RF stress (not narrative-only). |
| `apply_action_to_kpis` | Transparent KPI nudges from chosen action on scenario-derived bases. |

**Sourced vs assumed defaults:** `docs/SCENARIO_ENGINE_ASSUMPTIONS.md`.

## Site modeling (recognizable anchors)

| Capability | Type |
|------------|------|
| **Multi-vertex extruded footprints** per anchor (City Hall, Library, WSLA) | **Built-in** in `gary_site_geometry.py`; optional override via `configs/wireless_scene/site_footprints.json` |
| **Site outline accents** + **emoji labels** | Distinguishes civic vs library vs school at a glance |
| **Optional `.glb` + ScenegraphLayer** | `configs/wireless_scene/site_models.json` + `assets/models/`; **not required**; falls back to footprints |

**Default behavior:** extruded **footprint model** for all three sites. **Optional 3D asset model** when a valid GLB path loads successfully.

## Current capabilities

| Capability | Type |
|------------|------|
| Multi-layer **radio scene** (buildings, gNB, halos, demand, IF, propagation stress, links) | **Proxies**; **demand / IF** scale with **engine** outputs; building masses use **realistic outlines** (not single rectangles) |
| **Triple legend** (glyph / wireless stack / O-RAN mapping) | **Implemented** |
| **Propagation / Coverage** table + **Plotly bar chart** | **Proxy**; inputs include **coverage_pressure** / coexistence |
| **Closed-loop controller** | **Computed** action + KPI shift; **not** a live RIC |
| Detector in loop | **`evaluate()` on synthetic demo IQ** only |
| Drop zones + **JSON summaries** | `load_deepmimo_scenario_summary`, `load_sionna_propagation_summary`, `load_aerial_overlay_summary` in `simulation_integration_hooks.py` |

## Proxy vs integration-ready

- **Proxies:** halos, links, IF, orange stress disks, propagation scores, KPIs after heuristic action mapping.
- **Optional external sim:** summaries load **only** if JSON files exist; otherwise UI states **not loaded** + expected paths/schema hints.
- **Judged detector:** unchanged by extension hooks unless you explicitly integrate them into the submission package.

## Docs

- `docs/SIMULATION_BACKBONE_PLAN.md` — pillar detail + future compute/accounts  
- `docs/MICROTWIN_REALISM_PLAN.md` — UI behavior + loop diagram  
- `docs/SCENARIO_ENGINE_ASSUMPTIONS.md` — public defaults vs scenario assumptions  
- `docs/SITE_MODELING_PLAN.md` — footprint JSON, GLB paths, fallback rules  

## Run locally

```bash
streamlit run apps/streamlit_app.py
```

**Accounts:** None for the app. **Aerial / Omniverse** are optional **future** toolchain dependencies.

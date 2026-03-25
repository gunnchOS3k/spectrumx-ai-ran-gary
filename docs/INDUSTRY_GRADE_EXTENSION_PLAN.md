# Industry-grade Gary extension — capabilities, proxies, and next scaling

Three buckets:

1. **Core judged submission** — SpectrumX DAC detector evaluated **offline** on competition-style data; **raw competition IQ is not** shipped inside Streamlit.
2. **Completed extension** — Gary **digital-twin wireless scene** + **`gary_scenario_engine`** (population/devices/pressures) + **closed-loop AI-RAN scene controller** (**RIC-style**, **Near-RT / Non-RT–inspired**) in `_render_judge_gary_micro_twin_3d`.
3. **Simulation / deployment backbone (scaling path)** — DeepMIMO, Sionna RT, **AODT** (Aerial Omniverse Digital Twin: city/site-scale twin **target**), **pyAerial** bridge (PHY integration **path**), **Aerial Data Lake** / OTA evidence (**OTA-ready target**). Optional **JSON / manifest** drop zones; UI shows **provenance tiers** (`simulation_provenance.py`: proxy-only, demo summary, simulation export, installer-ready, integration stub, OTA-backed) — never blurred with the judged core.

**Roadmap:** `docs/AERIAL_STACK_ROADMAP.md` maps “must add now / next / later” to concrete repo paths.

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

## Occupancy & device visualization (scene)

| Layer / element | Tied to engine | Notes |
|-----------------|----------------|-------|
| **HeatmapLayer** (people / devices) | `people_count`, `active_ip_devices`, `active_control_devices` | Sampled points **inside** footprints; **weights** aggregate thousands of occupants. |
| **ScatterplotLayer** clusters | Cohort splits (students/staff, workers/visitors, patron zones) | Disk **radius** scales with counts — **not** one glyph per person. |
| **PathLayer** foot traffic | Preset + calendar **activity mode** | **Static** illustrative paths (ingress, circulation, queues). |
| **TextLayer** site summaries | Rounded people + active devices | Screenshot-friendly **aggregate** labels. |
| **Controller metrics** | Same `SiteScenarioState` | Explicit **total_device_count**, **active_device_count**, **people_count**, **traffic_demand_score**. |

Configs: `configs/wireless_scene/occupancy_profiles.json`, `device_profiles.json`, `movement_profiles.json`. Details: `docs/OCCUPANCY_VISUALIZATION_PLAN.md`.

## Current capabilities

| Capability | Type |
|------------|------|
| Multi-layer **radio scene** (buildings, gNB, halos, demand, IF, propagation, links, **people/device heatmaps**, **foot-traffic paths**, **aggregate clusters**) | **Proxies** + **aggregated** occupancy viz; **demand / IF** scale with **engine** outputs; footprints are **site-specific outlines** |
| **Triple legend** (glyph / wireless stack / O-RAN mapping) | **Implemented** |
| **Propagation / Coverage** table + **Plotly bar chart** | **Proxy**; inputs include **coverage_pressure** / coexistence |
| **Closed-loop controller** | **Computed** action + KPI shift; **not** a live RIC |
| Detector in loop | **`evaluate()` on synthetic demo IQ** only |
| Drop zones + **JSON / GeoJSON summaries** | `load_deepmimo_scenario_summary`, `load_sionna_propagation_summary`, `load_aerial_overlay_summary`, `load_pyaerial_bridge_status`, `load_ota_evidence_status` in `simulation_integration_hooks.py` — **Loaded** only after parse+validation; Sionna GeoJSON → optional **`layer-sionna-coverage-geojson`** on the map |
| Bundled demo exports | `examples/simulation_exports/deepmimo/`, `sionna_rt/`, `aerial_omniverse/`, `pyaerial_bridge/` — **fallback** when `data/*` empty/invalid; labeled per **provenance** in UI |
| **pyAerial bridge** | `src/edge_ran_gary/pyaerial_bridge/` + `docs/PYAERIAL_BRIDGE.md` — **integration-ready**; optional `data/pyaerial_bridge/bridge_manifest.json` |
| **OTA / Data Lake** | `src/edge_ran_gary/ota_data_interface.py`, `schemas/aerial_data_lake/`, `docs/DATA_LAKE_SCHEMA.md` — **OTA-ready target** until manifests exist |
| **Mobility / AODT hooks** | `configs/wireless_scene/mobility_flow_profiles.json`, `aodt_scene_hooks.example.yaml`, `gary_mobility_profiles.py` — **structured placeholders**; not a full mobility simulator in-app |

## Proxy vs integration-ready vs OTA target

- **Proxies:** halos, links, IF, orange stress disks, propagation scores, **static foot-traffic paths**, KPIs after heuristic action mapping. **Aggregates:** heatmaps / cluster disks (not per-person geometry).
- **Optional external sim:** summaries load **only** if JSON files exist; otherwise UI states **not loaded** + expected paths/schema hints. Each loader result includes **`provenance_tier` / `provenance_label`** (`attach_provenance_tier`).
- **AODT:** **Next scaling** for terrain, vegetation, materials, mobility, and dynamic scatterers when Omniverse / AI Aerial tooling runs; manifests in-repo are **integration-ready**, not full scene solves.
- **pyAerial / Aerial CUDA RAN:** **Bridge + operational target** language only; pyAerial is **not** required to run the app.
- **Data Lake:** **Evidence path** for OTA RF fields supporting calibration / retraining / twin replay; **OTA-backed** tier only when `data/ota_evidence/ota_lake_manifest.json` validates.
- **Judged detector:** unchanged by extension hooks unless you explicitly integrate them into the submission package.

## Docs

- `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md` — export scripts, NGC env vars (**names only**), status semantics  
- `docs/SIMULATION_BACKBONE_PLAN.md` — pillar detail + future compute/accounts  
- `docs/MICROTWIN_REALISM_PLAN.md` — UI behavior + loop diagram  
- `docs/SCENARIO_ENGINE_ASSUMPTIONS.md` — public defaults vs scenario assumptions  
- `docs/SITE_MODELING_PLAN.md` — footprint JSON, GLB paths, fallback rules  
- `docs/OCCUPANCY_VISUALIZATION_PLAN.md` — people / device / movement map layers  
- `docs/AERIAL_STACK_ROADMAP.md` — civic-scale 6G stack roadmap (repo paths)  
- `docs/EXTERNAL_RUNTIME_GAPS.md` — Streamlit vs GPU/lab/cloud (honest boundaries)  
- `docs/AODT_EXPORT_CHECKLIST.md` — full AODT vs export manifests  
- `docs/ARC_OTA_RUNTIME_TARGET.md` — ARC-OTA framing (**not active**)  
- `docs/DATA_LAKE_SCHEMA.md` — OTA capture schema and detector/twin mapping  
- `docs/PYAERIAL_BRIDGE.md` — pyAerial / PHY integration path  

## Run locally

```bash
streamlit run apps/streamlit_app.py
```

**Accounts:** None for the app itself. **NVIDIA AI Aerial / Omniverse** full pipelines typically require **NVIDIA tooling**, **GPU**, and often **NVIDIA account** / **6G Developer Program** access — optional **manifest JSON** under `data/aerial_omniverse/` is documented in `docs/SIMULATION_BACKBONE_PLAN.md`.

# Gary Micro-Twin — realism & controller loop plan

## Industry-grade O-RAN / AI-RAN pass (current Streamlit state)

The **Completed Research Extension** tab (`_render_judge_gary_micro_twin_3d` in `apps/streamlit_app.py`) is framed as a **conference-grade wireless systems demo** with strict separation:

| Bucket | What it is |
|--------|------------|
| **Core judged** | SpectrumX DAC detector — evaluated **offline** on competition-style data; **raw competition IQ not** embedded in Streamlit |
| **Completed extension** | Gary **digital-twin radio scene** + **scenario engine** (people/devices/load) + **closed-loop AI-RAN scene controller** (**RIC-style**, **Near-RT / Non-RT–inspired** abstraction) |
| **Next scaling** | DeepMIMO, Sionna RT, NVIDIA **AI Aerial / Omniverse** — JSON **summary loaders** + drop zones; physics sims **not** implied active unless files load |

### Scenario engine (formulas — high level)

Implemented in `src/edge_ran_gary/gary_scenario_engine.py`:

- **People:** site-specific (enrollment × attendance, staff presence, library baseline × occupancy ratio, city workers + visitors).
- **Devices:** IP and control counts from per-site device models; **active** counts = provisioned × **concurrency** (preset + manual traffic stress).
- **Traffic demand score (0–1):** `tanh`-shaped normalize of active IP vs a site reference + manual stress.
- **Coexistence pressure:** blend of traffic demand, RF environment slider, occupancy prior, control-to-IP ratio.
- **Coverage pressure:** active IP normalized by people proxy.
- **Fairness / community priority:** site-type bias + traffic (education/library weighted higher).
- **Propagation proxy bundle:** height + RF stress + **coverage_pressure** + **coexistence_pressure** → coverage / challenge / penetration **proxies** (not a solver).

**Public-grounded defaults vs assumptions:** see `docs/SCENARIO_ENGINE_ASSUMPTIONS.md`.

### Site geometry (footprints & optional 3D)

- **Built-in:** `src/edge_ran_gary/gary_site_geometry.py` supplies **multi-vertex footprints** (not generic cubes) for City Hall, Library, and West Side Leadership Academy, with **site-specific outline colors** and **map labels** (emoji + short name).
- **Optional JSON:** `configs/wireless_scene/site_footprints.json` overrides polygons/heights; `configs/wireless_scene/site_models.json` can point to **`assets/models/*.glb`** for **`ScenegraphLayer`**.
- **Fallback:** If no GLB or loader failure → **extruded footprint** only (app always runs).
- **Docs:** `docs/SITE_MODELING_PLAN.md` — schema, paths, local vs cloud caveats.

### Implemented in the extension UI

- **Guided demo strip (6 steps):** site → **scenario preset** → **wireless scene** (engine-driven) → **propagation / coverage** → **closed-loop controller** → KPIs.
- **Sources & assumptions** expander: tabulates defaults (sourced vs assumption).
- **Anchor strip + focus card:** community function, geometry mode (footprint vs 3D asset), footprint provenance note.
- **3D wireless scene (pydeck):** **extruded site footprints** (and optional **ScenegraphLayer**); **people + device HeatmapLayers** (aggregated samples); **PathLayer** foot-traffic **proxies**; **cluster disks** for cohorts / device load; **summary text** per site (elevated **TextLayer** + **billboard** + **outline** for contrast); **coverage halos**; violet **demand** radii ∝ **traffic_demand_score**; IF + propagation + gNB (**proxies**); optional **Sionna** **`GeoJsonLayer`** (`layer-sionna-coverage-geojson`) when `coverage_grid.geojson` validates; view **centered on anchor bbox**.
- **Triple legend:** glyph · wireless-layer stack · O-RAN **conceptual** mapping (**RIC-style controller**, not “full RT RIC”) — each column wrapped in a **light contrast card** for dark Streamlit themes.
- **Propagation / Coverage:** proxy table + bar chart; driven by scenario **pressures** + RF slider.
- **Closed-loop controller:** explicit **state vector** → **`select_closed_loop_action`** → **`apply_action_to_kpis`** (six candidate actions including **rebalance service**).
- **Simulation backbone:** cards + expander reuse the **same** early **`load_*_summary`** results — **`status_label`** (**Loaded (simulation export)** vs **Loaded (demo summary)** vs **Not loaded**), paths, optional JSON in expander (see **`docs/SIMULATION_BACKBONE_PLAN.md`**).
- **Simulation data sources + summaries:** buttons for **demo-only** (`examples/simulation_exports/*` only) vs **data/ first** with demo fallback; concise Cloud note (runtime `data/` not persistent); Sionna **coverage overlay** called out when GeoJSON validates; **Aerial** can show **Access confirmed / installer-ready** from `access_summary.json` (no secrets) via `scripts/check_ngc_access.py` — see `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`.
- **Map text:** building **TextLayer** uses short **plain names** (`map_label_scene`, no emoji) at **higher Z**; occupancy summary glyphs shortened (`Np · Nd` style) and elevated **Z** to reduce ground clutter.
- **Drop zones:** `data/deepmimo/`, `data/sionna_rt/`, `data/aerial_omniverse/`, `configs/wireless_scene/`, `configs/ric/` (+ legacy `data/simulation/*`). See `docs/SIMULATION_BACKBONE_PLAN.md`.

### Judge Mode (tour-level)

- **Why judges can trust this — evidence** (three buckets).
- **Why this aligns with AI-RAN / O-RAN** compact card (**Near-RT / Non-RT–inspired** language; no live E2 claim).

### What is proxy vs integration-ready

| Element | Status |
|--------|--------|
| Map layers (halos, links, IF, demand) | **Proxies**; demand/IF scale with **scenario engine** |
| Propagation table + bar chart | **Proxy abstraction** tied to engine pressures |
| Detector in controller loop | **`evaluate()` on synthetic demo IQ** only in-app |
| DeepMIMO / Sionna / Aerial | **JSON summary optional**; honest **not loaded** if absent |
| KPIs | **Scenario bases + action deltas** (`apply_action_to_kpis`) — not field measurements |

## Loop diagram

```text
Scenario inputs (preset, time, RF sliders, overrides)
        ↓
Scenario engine  →  SiteScenarioState (people, devices, pressures, fairness)
        ↓
Wireless scene map (multi-layer proxies scale with state)
        ↓
Sensing stand-in: evaluate() on synthetic IQ  →  detector belief
        ↓
Closed-loop AI-RAN scene controller: select_closed_loop_action
        ↓
KPI feedback: apply_action_to_kpis (coverage, coexistence, fairness, energy, continuity)
```

## References

- `src/edge_ran_gary/gary_site_geometry.py` — built-in footprints, JSON merge, optional GLB resolution  
- `src/edge_ran_gary/gary_occupancy_visualization.py` — occupancy / device / foot-traffic pydeck payloads  
- `src/edge_ran_gary/gary_scenario_engine.py` — scenario math + controller policy helpers  
- `src/edge_ran_gary/simulation_integration_hooks.py` — drop-zone status + JSON summary loaders  
- `docs/SCENARIO_ENGINE_ASSUMPTIONS.md` — sourced vs assumed counts  
- `docs/SITE_MODELING_PLAN.md` — footprint / 3D asset config  
- `docs/OCCUPANCY_VISUALIZATION_PLAN.md` — aggregated people/device/movement visualization  
- `docs/SIMULATION_BACKBONE_PLAN.md` — backbone detail + future compute/accounts  
- `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md` — capabilities summary  

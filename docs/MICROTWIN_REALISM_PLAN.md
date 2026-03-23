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

### Implemented in the extension UI

- **Guided demo strip (6 steps):** site → **scenario preset** → **wireless scene** (engine-driven) → **propagation / coverage** → **closed-loop controller** → KPIs.
- **Sources & assumptions** expander: tabulates defaults (sourced vs assumption).
- **3D wireless scene (pydeck):** buildings; **coverage halos** tied to coverage proxy + **coverage pressure**; violet **demand** radii ∝ **traffic_demand_score**; IF disks/footprints ∝ **max coexistence** across anchors; orange **propagation stress**; gNB / links (**proxies**).
- **Triple legend:** glyph · wireless-layer stack · O-RAN **conceptual** mapping (**RIC-style controller**, not “full RT RIC”).
- **Propagation / Coverage:** proxy table + bar chart; driven by scenario **pressures** + RF slider.
- **Closed-loop controller:** explicit **state vector** → **`select_closed_loop_action`** → **`apply_action_to_kpis`** (six candidate actions including **rebalance service**).
- **Simulation backbone:** cards + expander with **`load_*_summary`** — **loaded** path + JSON preview, or **not loaded** + expected filenames/schema hint.
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

- `src/edge_ran_gary/gary_scenario_engine.py` — scenario math + controller policy helpers  
- `src/edge_ran_gary/simulation_integration_hooks.py` — drop-zone status + JSON summary loaders  
- `docs/SCENARIO_ENGINE_ASSUMPTIONS.md` — sourced vs assumed counts  
- `docs/SIMULATION_BACKBONE_PLAN.md` — backbone detail + future compute/accounts  
- `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md` — capabilities summary  

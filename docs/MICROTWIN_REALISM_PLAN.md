# Gary Micro-Twin — realism & controller loop plan

## Industry-grade O-RAN / AI-RAN pass (current Streamlit state)

The **Completed Research Extension** tab (`_render_judge_gary_micro_twin_3d` in `apps/streamlit_app.py`) is framed as a **conference-grade wireless systems demo** with strict separation:

| Bucket | What it is |
|--------|------------|
| **Core judged** | SpectrumX DAC detector on **official competition data** (offline); not loaded in this tab |
| **Completed extension** | Gary **digital-twin radio scene** + **Near-RT RIC–style xApp abstraction** (policy demo, proxies) |
| **Next scaling** | DeepMIMO, Sionna RT, NVIDIA **AI Aerial / Omniverse** — **integration-ready** drop zones + stubs |

### Implemented in the extension UI

- **Guided demo strip (6 steps):** site → scenario → **wireless scene** → **propagation / coverage** → **RIC state → action** → KPIs.
- **Evidence cards:** **Why this is research-grade now** + **Why this matters for 6G research & admissions** (detector, twin, RIC story, simulation discipline).
- **3D wireless scene (pydeck):** buildings; **light-green coverage halos** (size ∝ coverage proxy); cyan **serving links**; violet **demand**; orange **propagation stress**; red **IF** disks + polygons; blue **gNB** (RU abstraction); labels.
- **Triple legend:** **Map / glyph** · **Wireless-layer stack (L1–L5 semantics)** · **O-RAN conceptual mapping** (RU, O1 stand-in, Near-RT / xApp).
- **Propagation / Coverage:** dedicated section — proxy table, focus metrics, **Plotly grouped bar chart** (screenshot panel); explicit **not DeepMIMO / not Sionna / not Aerial** labeling.
- **Near-RT RIC–style loop:** state **ingestion** → **belief/policy** → **action selection** → **KPI feedback**; candidate actions; policy output callout.
- **Simulation backbone:** **three cards** — **DeepMIMO** | **Sionna RT** | **NVIDIA AI Aerial / AODT** — each with contributes / connects / **honest status** (stub vs active).
- **Drop zones:** `data/deepmimo/`, `data/sionna_rt/`, `data/aerial_omniverse/`, `configs/wireless_scene/`, `configs/ric/` (+ legacy `data/simulation/*`). See `docs/SIMULATION_BACKBONE_PLAN.md`.

### Judge Mode (tour-level)

- **Why judges can trust this — evidence** (three buckets).
- **Why this aligns with AI-RAN / O-RAN** compact card (Near-RT RIC narrative; no live E2 claim).

### What is proxy vs integration-ready

| Element | Status |
|--------|--------|
| Map layers (halos, links, IF, demand) | **Implemented proxy** |
| Propagation table + bar chart | **Implemented proxy abstraction** |
| Detector in RIC loop | **Synthetic demo IQ** only in-app |
| DeepMIMO / Sionna / Aerial driving UI | **Not yet** — stubs + directory scan |
| KPIs | **Proxy only** |

## Loop diagram

```text
Scenario  →  twin state (site, demand, RF stress proxies)
        ↓
Wireless scene map (multi-layer proxies)
        ↓
Sensing stand-in: evaluate() on synthetic IQ  →  belief inputs
        ↓
Near-RT RIC–style policy: discrete action
        ↓
KPI feedback (coverage, coexistence, equity, energy, continuity)
```

## References

- `src/edge_ran_gary/simulation_integration_hooks.py` — drop-zone status + stubs  
- `docs/SIMULATION_BACKBONE_PLAN.md` — backbone detail + future compute/accounts  
- `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md` — capabilities summary  

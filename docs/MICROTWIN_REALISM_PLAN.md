# Gary Micro-Twin — realism & controller loop plan

## Industry-grade extension pass (current Streamlit state)

The **Completed Research Extension** tab (`_render_judge_gary_micro_twin_3d` in `apps/streamlit_app.py`) is structured as a **research-grade wireless digital-twin demo** with strict honesty about proxies:

- **Guided walkthrough (6 steps):** Primary path for nontechnical viewers — site → scenario → detector → propagation → controller → KPIs.
- **Evidence panels:** **Why this is research-grade now** and **Why this matters for 6G PhD research** (cards; core judged path called out as separate).
- **3D wireless scene (pydeck):** Extruded footprints; **cyan PathLayer** = gNB→site **serving-link proxy** (tooltip LOS fraction = **indicator only**); violet demand; red interference **disks + polygon footprints**; blue gNB markers; labels.
- **Dual legends:** **Map legend** (colors/symbols) + **Radio-scene legend** (semantics: proxy vs real asset DB).
- **Radio Environment / Propagation View:** Dedicated panel; **all metrics labeled as current proxy propagation view**; per-site table (pandas) or captions; **next scaling** callout for Sionna RT / DeepMIMO.
- **AI-RAN controller:** Explicit **state** row (detector belief, site, demand, radio env score, coexistence risk); **prominent chosen action** (`st.success`); pipeline trace; **Why this is AI-RAN** explainer; five **proxy KPIs** (coverage, coexistence, energy, fairness/community benefit, service continuity).
- **Simulation backbone:** Three columns — **implemented now** / **DeepMIMO ready path** / **Sionna RT ready path**; JSON expander for directory status; stub loaders report **no overlay** until implemented.
- **Optional local assets expander:** Lists `submission_metrics.csv`, `final_report_figures.yaml`, `gary_micro_twin.yaml`, simulation dirs — **fails gracefully** if missing.

Core **SpectrumX DAC detector** scoring data and official IQ remain **out of band** for this tab.

## What is implemented vs proxy vs next scaling

| Element | Status |
|--------|--------|
| Extruded building footprints, heights, scenario-driven tint | **Implemented** (approximate coordinates) |
| gNB / demand / interference overlays (scatter + polygon IF regions + link paths) | **Implemented proxy** (not field-calibrated) |
| Propagation table / scores | **Simulated proxy** (deterministic; labeled in UI) |
| Live detector in loop | **Implemented** when `evaluate()` has run on **synthetic** Judge demo IQ |
| DeepMIMO / Sionna **outputs driving UI** | **Not yet** — directories + `simulation_integration_hooks.py` stubs only |
| KPI metrics | **Proxy only** |

## User → environment → controller → outcome loop

```text
Users & time/event scenario  →  demand & occupancy priors (proxy)
        ↓
Map: buildings + gNB + demand + interference + link segments (all partially proxy)
        ↓
Sensed spectrum (submission evaluate() on demo IQ only in-app)  →  belief + stress
        ↓
Controller: state vector → discrete action (hold / cautious TX / power / channel / prioritize)
        ↓
Outcome proxies: coverage, coexistence, fairness/community, energy, continuity
```

## Next research scaling path

1. **DeepMIMO** — channel / scenario artifacts under `data/simulation/deepmimo/`; wire `load_deepmimo_overlay_stub` to real parsers.
2. **Sionna RT** — GeoJSON / JSON under `data/simulation/sionna_rt/`; replace proxy propagation panel where appropriate.
3. **Controller replay** logs + static export for papers.

See **`docs/INDUSTRY_GRADE_EXTENSION_PLAN.md`** for hook details and file conventions.

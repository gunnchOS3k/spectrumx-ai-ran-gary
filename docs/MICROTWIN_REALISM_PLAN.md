# Gary Micro-Twin — realism & controller loop plan

## What changed in the Digital Twin Realism Pass (Streamlit)

The **Completed Research Extension** tab (`_render_judge_gary_micro_twin_3d` in `apps/streamlit_app.py`) was redesigned for **conference-demo** clarity and is presented as a **finished extension prototype** for judges:

- **3D map is primary:** Full-width pydeck scene first (after a compact scenario toolbar): extruded footprints + **violet demand disks** + **red interference proxies** + **blue gNB points** + labels.
- **Scenario toolbar:** Demand, occupancy prior, RF environment, **time context** (school / after hours / weekend), **event mode** (normal vs special event) — combined with **focus site** to adjust effective stress weights and map radii.
- **Per-site identity:** Building type, height, approximate footprint, and **why this site matters** in metric/cards (not a single generic `st.info`).
- **Radio environment row:** Four **always-visible** cards (gNB position, demand hotspots, interference zones, low-7 GHz LOS / penetration / blockage **proxies**) with explicit **`implemented proxy`** vs **`future integration`** tags.
- **Users at this site:** Persona **cards** (City Hall / Library / West Side) driven by the selected anchor.
- **RAN controller:** **Five-column pipeline** (Sense → Belief → Site context → Action → KPI) plus a **candidate-actions** row; selected action includes a **one-line reason** tied to scenario + site.
- **KPI row:** Five **proxy** metrics — coverage, coexistence, community benefit, energy/efficiency, service continuity — all **honestly labeled** as non-measured.
- **Nontechnical RF panel:** Short **“How low-7 GHz … interacts with the site”** explanation.
- **6G roadmap:** Three **column cards** (DeepMIMO, Sionna RT, beam/coverage UI) as **future integration** — no overclaim.
- **Heavy expanders removed** for the old “Layer 1–5” text wall; layers are now **visible cards + map overlays**.
- Duplicate **Research-Grade 6G** markdown block under the judge tab was **removed** (content folded into the roadmap panel inside the twin renderer).

Core **SpectrumX DAC detector** routing, submission adapter, and **judged vs future work** separation elsewhere in the app were **not** reworked in this pass.

## What is implemented vs proxy vs next scaling

| Element | Status |
|--------|--------|
| Extruded building footprints, heights, scenario-driven tint | **Implemented** (approximate coordinates) |
| gNB / demand / interference **scatter overlays** on map | **Implemented proxy** (storytelling geometry, not field-calibrated) |
| LOS / penetration / blockage **numbers** in radio card | **Simulated proxy** (deterministic from scenario + building height; proxy model) |
| Live detector output in pipeline | **Implemented** when `evaluate()` has run (Judge demo IQ); else “unknown” / hold |
| DeepMIMO / Sionna RT / beam–coverage maps | **Future integration** (documented only in UI + this doc) |
| KPI metrics | **Proxy only** (not drive-test or field data) |

## User → environment → controller → outcome loop

```text
Users & time/event scenario  →  demand & occupancy priors (proxy)
        ↓
Map: buildings + gNB + demand disks + interference disks
        ↓
Sensed spectrum (submission evaluate() on demo IQ)  →  belief + stress
        ↓
Controller selects among: hold / cautious TX / reduce power / switch channel / prioritize site
        ↓
Outcome proxies: coverage, coexistence, equity story, energy, continuity
```

The Micro-Twin and 3D paths are a **completed research extension** (non-scoring), separate from the official SpectrumX DAC scoring basis.

## Next research scaling path (future integration)

1. Optional **DeepMIMO**-style channel export per anchor footprint (behind flag).
2. **Sionna RT** (or similar) for real blockage / coverage tiles; feed optional heatmap layer in pydeck.
3. **Controller replay** log + static figure export for papers.
4. Calibrate proxies against a **published link-budget worksheet** (teaching only).

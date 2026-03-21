# Industry-grade Gary extension — capabilities, proxies, and next scaling

Three buckets:

1. **Core judged submission** — SpectrumX DAC detector on **official competition data** (offline; **not** loaded in Streamlit).
2. **Completed extension** — Gary **digital-twin wireless scene** + **Near-RT RIC–style (xApp-like)** control abstraction in `_render_judge_gary_micro_twin_3d`.
3. **Next realism scaling** — DeepMIMO, Sionna RT, **NVIDIA AI Aerial / Omniverse** — **integration-ready** drop dirs + stubs until parsers exist.

## Current capabilities

| Capability | Type |
|------------|------|
| Multi-layer **radio scene** (buildings, gNB, halos, demand, IF, propagation stress, links) | **Implemented proxies** (labeled) |
| **Triple legend** (glyph / wireless stack / O-RAN mapping) | **Implemented** |
| **Propagation / Coverage** table + **Plotly bar chart** | **Implemented proxy abstraction** |
| **RIC loop** wording: ingestion → belief/policy → action → KPI feedback | **Demo**; not a live RIC |
| Detector in loop | **`evaluate()` on synthetic demo IQ** only |
| Drop zones: `data/deepmimo/`, `data/sionna_rt/`, `data/aerial_omniverse/`, `configs/wireless_scene/`, `configs/ric/` (+ legacy `data/simulation/*`) | **Hooks** in `simulation_integration_hooks.py` |

## Proxy vs integration-ready

- **Proxies:** halos, links, IF, orange stress disks, propagation scores, discrete actions, KPIs.
- **Not claimed active:** DeepMIMO, Sionna RT, Aerial in the **judged** detector unless explicitly wired in code.

## Docs

- `docs/SIMULATION_BACKBONE_PLAN.md` — pillar detail + future compute/accounts  
- `docs/MICROTWIN_REALISM_PLAN.md` — UI behavior  

## Run locally

```bash
streamlit run apps/streamlit_app.py
```

**Accounts:** None for the app. **Aerial / Omniverse** are optional **future** toolchain dependencies.

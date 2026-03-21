# Gary Micro-Twin — realism & controller loop plan

## Current state (Streamlit)

- **Synthetic IQ:** `GaryMicroTwin.generate_samples_per_zone` produces zone-labeled windows with metadata (`label`, `zone_id`, `snr_db`, `cfo_hz`, `num_taps`, `signal_type`, etc.). Noise-only rows use `signal_type: null` / NaNs for RF parameters.
- **3D site view:** Approximate building footprints (City Hall, Library, West Side Leadership Academy) rendered with **pydeck** extruded polygons; scenario sliders drive **risk tint** (storytelling proxy only).
- **Layered model UI:** Five layers (Site → Users → Radio → Controller → Outcome) are documented in-app with tags: **implemented**, **simulated proxy**, **future integration**.
- **RAN controller demo:** Uses **live** `evaluate()` output from the selected submission package on **Judge Mode synthetic demo IQ** plus scenario sliders to pick a **candidate action** and **proxy KPIs** (coverage, coexistence, fairness). Not field-validated.

## Next realism upgrades (ordered)

1. **Radio Layer:** Optional import path for **DeepMIMO**-style channel draws or **Sionna RT** ray-tracing hooks (behind feature flags); keep UI honest about “not driving judged detector until integrated.”
2. **User / demand Layer:** Tie occupancy priors to time-of-day presets or simple stochastic models; still synthetic.
3. **Controller Layer:** Replace rule table with logged policy from recorded detector statistics; optional small RL storyboard (research only).
4. **Outcome Layer:** Calibrate proxy KPIs against simple link-budget spreadsheets for teaching, not claims of measured coverage.

## User → environment → controller → KPI loop

```text
Resident / student / visitor need  →  site context + demand
        ↓
Sensed spectrum (detector)  →  belief (occupied? interference stress?)
        ↓
RAN action (hold, power down, retune, prioritize site)
        ↓
Outcome proxies (coverage, coexistence, equity, energy)
```

All Micro-Twin and 3D paths remain **extension / future work**, not the official SpectrumX DAC scoring basis.

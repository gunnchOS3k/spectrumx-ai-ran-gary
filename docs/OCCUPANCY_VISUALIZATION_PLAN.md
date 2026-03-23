# Occupancy, devices & foot-traffic visualization (Gary micro-twin)

## Purpose

Make **people**, **devices**, and **movement** **visible** in the pydeck scene for nontechnical viewers, without rendering thousands of 3D avatars. The **Completed Research Extension** uses **aggregated** and **representative** graphics tied to `gary_scenario_engine.SiteScenarioState`.

**Core judged** SpectrumX detector and **next scaling** (DeepMIMO / Sionna) are unchanged categories.

## What is individual vs aggregated vs proxy

| Element | Type | Meaning |
|--------|------|--------|
| **HeatmapLayer (people)** | **Aggregated** | Sampled points **inside** each footprint; **weight** ∝ scenario `people_count` (distributed across samples). |
| **HeatmapLayer (devices)** | **Aggregated** | Samples weighted by **active IP** + **control** radios (control up-weighted slightly). |
| **Scatterplot “clusters”** | **Aggregated** | Large translucent disks; **radius** scales with cohort size (students vs staff, workers vs visitors, patron zones). **Not** one disk per person. |
| **Hero markers** (small dots) | **Representative** | Fixed small glyphs — **illustrative** only; count does not match dot count. |
| **PathLayer (flows)** | **Proxy** | **Static** polylines; shape varies by **activity mode** (normal day, peak, after hours, event, emergency). **Not** GPS or simulation traces. |
| **TextLayer (summary)** | **Aggregated** | Short string per site: rounded **people** and **active devices**. |
| **Demand / coexistence / KPI** | **Engine + policy** | Already driven by scenario state; visuals **align** with the same numbers. |

## Activity modes (foot-traffic styling)

Derived from **preset** (`normal_day`, `peak_day`, `after_hours`, `emergency_special`), **calendar** (`School hours`, …), and **special event** unless overridden by `movement_profiles.json` → `force_activity_mode`.

Examples:

- **class_change_peak** — longer / thicker school paths during peak + school hours.
- **after_hours** — minimal school paths; different library/civic emphasis.
- **event_surge** — special event flag; stronger program-room path at library.

## Optional configs

| File | Role |
|------|------|
| `configs/wireless_scene/occupancy_profiles.json` | `heatmap_people_samples_per_site`, `heatmap_people_radius_pixels` |
| `configs/wireless_scene/device_profiles.json` | Device heatmap sample count / radius |
| `configs/wireless_scene/movement_profiles.json` | `force_activity_mode` (optional pin) |

If files are missing or keys omitted, **built-in defaults** apply.

## Layer IDs (pydeck)

Unique `id` fields include:

- `layer-occupancy-people-heatmap`
- `layer-occupancy-device-heatmap`
- `layer-foot-traffic-flows`
- `layer-occupancy-people-clusters`
- `layer-occupancy-device-clusters`
- `layer-occupancy-hero-markers`
- `layer-occupancy-summary-labels`

## Fallback

If `gary_occupancy_visualization` fails to import or throws, the app **omits** these layers and keeps buildings + wireless stack only.

## Code

- `src/edge_ran_gary/gary_occupancy_visualization.py` — data builders
- `apps/streamlit_app.py` — layer assembly + legend + controller metrics

# Architecture decision summary

Short, review-ready rationale aligned with post-project honesty.

| Decision | Rationale |
|----------|-----------|
| **Controller labeled rule-based baseline (RIC-style abstraction)** | Shipped Streamlit extension uses **detector-conditioned** heuristic actions and KPI deltas (`select_closed_loop_action` / `apply_action_to_kpis`). Trained RL or contextual-bandit policies are **not** claimed as current. |
| **Provenance vocabulary (six evidence terms, three execution surfaces)** | Prevents “demo looks like production PHY” confusion; separates proxy/demo, loaded exports, installer-ready, external-runtime, and OTA-backed claims. See [`docs/PROVENANCE_LEGEND.md`](../PROVENANCE_LEGEND.md). |
| **Streamlit = presentation + manifest literacy, not full external runtime** | Hosted app loads manifests and synthetic/demo IQ; **AODT / full Sionna RT / pyAerial execution / OTA capture** remain **external** targets or lab workflows. See [`docs/EXTERNAL_RUNTIME_GAPS.md`](../EXTERNAL_RUNTIME_GAPS.md). |
| **AODT / pyAerial / OTA as external targets** | Keeps adoption narrative honest while still showing **integration seams** (hooks, bridge types, manifest paths) for future validation loops. |
| **Gary limited to three anchors in current scope** | **Gary City Hall**, **Gary Public Library & Cultural Center**, **West Side Leadership Academy** bound the digital-twin narrative and scenario UI; broader geography is out of current scope. |

[← UML README](README.md)

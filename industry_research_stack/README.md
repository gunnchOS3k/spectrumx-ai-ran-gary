# Industry / Research-Grade Tooling Stack

| Category | Examples | This repo |
|----------|----------|-----------|
| Open optional | DeepMIMO configs, O-RAN KPI schema, ns-3 export stubs | Adapters in `src/*/tool_adapters/` |
| GPU optional | NVIDIA Sionna | `scripts/check_sionna_available.py` — skips if missing |
| Access required | NVIDIA Aerial / Omniverse DT, commercial lab tools | Docs + neutral JSON exports only |

**Default path:** smoke exports in `results/tool_exports/` — preparation artifacts, not validated RAN results.

**Evidence:** Level 1 smoke unless a real backend run is documented on `main`.

See `NON_AFFILIATION_NOTICE.md`, `OPTIONAL_BACKENDS.md`, `EVIDENCE_UPGRADE_PATH.md`.

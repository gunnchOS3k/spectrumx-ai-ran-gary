# pyAerial / Aerial CUDA-Accelerated RAN bridge

## Honest scope

| Item | In this repo |
|------|----------------|
| **pyAerial** Python package | **Optional**; `describe_pyaerial_environment()` tries `import pyaerial` |
| **cuPHY / CUDA RAN** | **External** NVIDIA Aerial stack; **operational target**, not bundled |
| **Detector → PHY hints** | **Frozen dataclass** `PHYControlPlaneHints` + `.as_public_dict()` in `phy_interface.py` |
| **cuMAC abstraction** | **Frozen dataclass** `CUMACSchedulerAbstraction` — not NVIDIA cuMAC |

## Code layout

- `src/edge_ran_gary/pyaerial_bridge/` — bridge package
- `notebooks/pyAerial_bridge_demo.ipynb` — optional walkthrough (probe + detector → PHY hints + cuMAC **abstraction**)
- `data/pyaerial_bridge/` — drop zone for `bridge_manifest.json` when you probe a lab environment
- `examples/simulation_exports/pyaerial_bridge/bridge_manifest.json` — **stub** manifest (demo tier)

## Conceptual dataflow

1. **Sensing:** SpectrumX-style IQ + binary occupancy from `evaluate()` (synthetic in Streamlit; official offline).
2. **Belief / policy:** Scenario engine + RIC-style controller in the completed extension.
3. **PHY hints:** `detector_to_phy_control_plane_hints()` returns **`PHYControlPlaneHints`** (OFDM / CE / timing / CFO / power **stories**) — **not** a wire protocol to a RU. Fields `relates_to_cuPHY` and `relates_to_aerial_cuda_ran` state external alignment only.

4. **cuMAC-style scheduling:** `cumac_scheduler_abstraction()` returns **`CUMACSchedulerAbstraction`** — narrative only, **not** cuMAC code.

## Evidence / execution vocabulary

Bridge manifests use the same **six evidence** and **three execution** terms as other hooks (`docs/PROVENANCE_LEGEND.md`). **pyAerial import OK** does **not** upgrade evidence to a full PHY run; execution remains **external-runtime-required** for real cuPHY.

## Installation path

When pyAerial is available in **your** conda/venv:

```python
from src.edge_ran_gary.pyaerial_bridge import describe_pyaerial_environment
print(describe_pyaerial_environment())
```

Until then, rely on **documentation + manifest stub** only.

## Relation to ARC-OTA

**ARC-OTA** and **runtime-target** framing: treat **Aerial CUDA-Accelerated RAN** as the credible **lab / field** execution target; this repo stays **integration-ready** and **OTA-ready** without claiming live OTA runs.

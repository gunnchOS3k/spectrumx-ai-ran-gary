# pyAerial / Aerial CUDA-Accelerated RAN bridge

## Honest scope

| Item | In this repo |
|------|----------------|
| **pyAerial** Python package | **Optional**; `describe_pyaerial_environment()` tries `import pyaerial` |
| **cuPHY / CUDA RAN** | **External** NVIDIA Aerial stack; **operational target**, not bundled |
| **Detector → PHY hints** | **Typed placeholders** in `pyaerial_bridge/phy_interface.py` |

## Code layout

- `src/edge_ran_gary/pyaerial_bridge/` — bridge package
- `notebooks/pyAerial_bridge_demo.ipynb` — optional walkthrough (probe + detector → PHY hints + cuMAC **abstraction**)
- `data/pyaerial_bridge/` — drop zone for `bridge_manifest.json` when you probe a lab environment
- `examples/simulation_exports/pyaerial_bridge/bridge_manifest.json` — **stub** manifest (demo tier)

## Conceptual dataflow

1. **Sensing:** SpectrumX-style IQ + binary occupancy from `evaluate()` (synthetic in Streamlit; official offline).
2. **Belief / policy:** Scenario engine + RIC-style controller in the completed extension.
3. **PHY hints (future):** `detector_to_phy_control_plane_hints()` suggests **OFDM mask story**, **channel estimation refresh bias**, **timing/CFO loop** narrative — **not** a wire protocol to a RU.

4. **cuMAC-style scheduling (abstraction):** `cumac_scheduler_abstraction()` documents how twin KPIs could **narrate** MAC objectives in an external Aerial stack — **not** cuMAC code.

## Installation path

When pyAerial is available in **your** conda/venv:

```python
from src.edge_ran_gary.pyaerial_bridge import describe_pyaerial_environment
print(describe_pyaerial_environment())
```

Until then, rely on **documentation + manifest stub** only.

## Relation to ARC-OTA

**ARC-OTA** and **runtime-target** framing: treat **Aerial CUDA-Accelerated RAN** as the credible **lab / field** execution target; this repo stays **integration-ready** and **OTA-ready** without claiming live OTA runs.

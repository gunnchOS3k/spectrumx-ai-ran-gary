"""
pyAerial / cuPHY-style PHY bridge (completed extension, **next scaling path**).

**Honest scope:** This package does **not** require NVIDIA pyAerial to be installed.
It documents how **detector beliefs** and **IQ features** could feed **OFDM / channel /
timing / CFO** PHY-side hooks when a lab stack is available.

**Judged detector:** unchanged; training and ``evaluate()`` remain on SpectrumX data offline.
"""

from src.edge_ran_gary.pyaerial_bridge.phy_interface import (
    PHYBridgeStatus,
    cumac_scheduler_abstraction,
    describe_pyaerial_environment,
    detector_to_phy_control_plane_hints,
    suggested_integration_paths,
)

__all__ = [
    "PHYBridgeStatus",
    "cumac_scheduler_abstraction",
    "describe_pyaerial_environment",
    "detector_to_phy_control_plane_hints",
    "suggested_integration_paths",
]

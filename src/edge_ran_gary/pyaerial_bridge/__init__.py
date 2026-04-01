"""
pyAerial / cuPHY-style PHY bridge (research extension, **external runtime** for real PHY).

**Honest scope:** This package does **not** require NVIDIA pyAerial. It documents how
detector beliefs could map to **conceptual** PHY/MAC hooks when a lab stack exists.

**Judged detector:** unchanged; training and ``evaluate()`` remain on SpectrumX data offline.
"""

from src.edge_ran_gary.pyaerial_bridge.phy_interface import (
    CUMACSchedulerAbstraction,
    PHYBridgeStatus,
    PHYControlPlaneHints,
    cumac_scheduler_abstraction,
    describe_pyaerial_environment,
    detector_to_phy_control_plane_hints,
    suggested_integration_paths,
)

__all__ = [
    "CUMACSchedulerAbstraction",
    "PHYBridgeStatus",
    "PHYControlPlaneHints",
    "cumac_scheduler_abstraction",
    "describe_pyaerial_environment",
    "detector_to_phy_control_plane_hints",
    "suggested_integration_paths",
]

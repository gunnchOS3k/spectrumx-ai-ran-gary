"""
PHY control-plane hints: maps high-level detector outputs to **conceptual** PHY knobs.

**Not** a live CUDA PHY. Use when integrating **Aerial CUDA-Accelerated RAN** or
**pyAerial** in a separate runtime (ARC-OTA / lab target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PHYBridgeStatus:
    """Whether optional pyAerial / Aerial PHY Python bindings are importable."""

    pyaerial_import_ok: bool
    import_error: Optional[str] = None
    note: str = field(
        default_factory=lambda: (
            "PHY bridge is **documentation + typed hints** unless pyAerial/Aerial stack is installed "
            "in your environment. See docs/PYAERIAL_BRIDGE.md."
        )
    )


def describe_pyaerial_environment() -> PHYBridgeStatus:
    try:
        import pyaerial  # type: ignore

        _ = pyaerial  # noqa: F841
        return PHYBridgeStatus(pyaerial_import_ok=True, import_error=None, note="pyAerial import succeeded.")
    except ImportError as e:
        return PHYBridgeStatus(pyaerial_import_ok=False, import_error=str(e)[:240])
    except Exception as e:  # pragma: no cover
        return PHYBridgeStatus(pyaerial_import_ok=False, import_error=f"{type(e).__name__}: {e}"[:240])


def detector_to_phy_control_plane_hints(
    occupancy_label: int,
    confidence: Optional[float],
    *,
    scenario_traffic_score: float = 0.5,
    rf_stress: float = 0.3,
) -> Dict[str, Any]:
    """
    Produce **research placeholders** for how a binary occupancy detector could
    inform PHY / MAC-style decisions in an AI-RAN loop.

    Keys are **semantic** (not wire-format to a real RU).
    """
    occ = int(occupancy_label) if occupancy_label in (0, 1) else -1
    conf_f = float(confidence) if confidence is not None else None
    cautious = (occ == 1) or (conf_f is not None and conf_f > 0.55) or (scenario_traffic_score > 0.65)

    hints: Dict[str, Any] = {
        "ofdm_subcarrier_mask_suggestion": "full_band" if not cautious else "prefer_guard_expansion_story",
        "channel_estimation_refresh_bias": "normal" if not cautious else "aggressive",
        "timing_track_loop_bandwidth": "nominal" if rf_stress < 0.45 else "tighten",
        "cfo_track_prior_hz": 0.0,
        "power_backoff_db_story": 0.0 if not cautious else min(6.0, 2.0 + 8.0 * rf_stress),
    }
    hints["declared_scope"] = (
        "Abstract control hints for **AI-RAN storytelling** and future pyAerial/cuPHY integration; "
        "**not** applied to hardware in this repo."
    )
    return hints


def suggested_integration_paths() -> List[str]:
    return [
        "NVIDIA **Aerial CUDA-Accelerated RAN** as operational RAN target (external install).",
        "**pyAerial** Python APIs for PHY experimentation when licensed/available.",
        "Export **IQ windows + labels** to `data/ota_evidence/` for **OTA capture replay** (see docs/DATA_LAKE_SCHEMA.md).",
    ]


def cumac_scheduler_abstraction() -> Dict[str, str]:
    """
    **Honest** MAC/scheduling vocabulary aligned with AI-RAN storytelling.

    **Not** NVIDIA cuMAC or any production MAC. Use only to explain how twin **priorities**
    could map to scheduler objectives in an external Aerial stack.
    """
    return {
        "scope": "Abstract **cuMAC-style** scheduling story: map coexistence + fairness KPIs to resource allocation narratives.",
        "relation_to_twin": "Gary scenario engine outputs (traffic, fairness, coverage pressure) are **inputs** to policy, not MAC code.",
        "execution": "Real **cuMAC / MAC** runs only in **external** Aerial or lab RAN targets (`docs/EXTERNAL_RUNTIME_GAPS.md`).",
    }

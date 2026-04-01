"""
PHY / MAC **abstractions** for AI-RAN research narratives.

**Declared scope:** No live **cuPHY**, **pyAerial** execution, or RU wire protocol in this module.
Real PHY runs only in **external** NVIDIA Aerial / lab targets (`docs/EXTERNAL_RUNTIME_GAPS.md`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PHYBridgeStatus:
    """Result of an optional ``import pyaerial`` probe (never faked)."""

    pyaerial_import_ok: bool
    import_error: Optional[str] = None
    declared_scope: str = (
        "Import probe only. A **false** value means pyAerial is absent; use an **external** Aerial runtime for real PHY."
    )
    note: str = field(
        default_factory=lambda: (
            "PHY bridge defaults to **typed hints** unless pyAerial is installed in your environment. "
            "See docs/PYAERIAL_BRIDGE.md."
        )
    )

    def as_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PHYControlPlaneHints:
    """
    **Conceptual** mapping from detector belief to PHY-side knobs (OFDM / CE / timing / CFO / power story).

    **Not** a 3GPP or vendor control-plane PDU. **Not** executed against hardware here.
    """

    ofdm_subcarrier_mask_suggestion: str
    channel_estimation_refresh_bias: str
    timing_track_loop_bandwidth: str
    cfo_track_prior_hz: float
    power_backoff_db_story: float
    declared_scope: str
    relates_to_cuPHY: str = (
        "Narrative alignment with **cuPHY-style** PHY in **external** Aerial CUDA-Accelerated RAN; **no** cuPHY API calls in-repo."
    )
    relates_to_aerial_cuda_ran: str = (
        "Operational stack target: **Aerial CUDA-Accelerated RAN** (external install)."
    )

    def as_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CUMACSchedulerAbstraction:
    """Honest **cuMAC-style** scheduling vocabulary (not NVIDIA cuMAC)."""

    scope: str
    relation_to_twin: str
    execution: str
    declared_scope: str = (
        "Abstract MAC/scheduling **story** for reviewers; **not** a scheduler implementation."
    )

    def as_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


def describe_pyaerial_environment() -> PHYBridgeStatus:
    try:
        import pyaerial  # type: ignore

        _ = pyaerial  # noqa: F841
        return PHYBridgeStatus(
            pyaerial_import_ok=True,
            import_error=None,
            note="pyAerial import succeeded in this kernel (PHY execution still **external** to this repo).",
        )
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
) -> PHYControlPlaneHints:
    """
    Map binary occupancy + scenario stress to **typed** PHY hint fields.

    Returns a **frozen dataclass**; use ``.as_public_dict()`` for JSON-like views.
    """
    occ = int(occupancy_label) if occupancy_label in (0, 1) else -1
    conf_f = float(confidence) if confidence is not None else None
    cautious = (occ == 1) or (conf_f is not None and conf_f > 0.55) or (scenario_traffic_score > 0.65)

    return PHYControlPlaneHints(
        ofdm_subcarrier_mask_suggestion="full_band" if not cautious else "prefer_guard_expansion_story",
        channel_estimation_refresh_bias="normal" if not cautious else "aggressive",
        timing_track_loop_bandwidth="nominal" if rf_stress < 0.45 else "tighten",
        cfo_track_prior_hz=0.0,
        power_backoff_db_story=0.0 if not cautious else min(6.0, 2.0 + 8.0 * rf_stress),
        declared_scope=(
            "Abstract control hints for **AI-RAN** documentation and future **pyAerial** integration; "
            "**not** applied to hardware in this repository."
        ),
    )


def suggested_integration_paths() -> List[str]:
    return [
        "NVIDIA **Aerial CUDA-Accelerated RAN** as operational RAN target (external install).",
        "**pyAerial** Python APIs for PHY experimentation when licensed/available in **your** GPU environment.",
        "Export **IQ windows + labels** to `data/ota_evidence/` for **OTA capture replay** (see docs/DATA_LAKE_SCHEMA.md).",
    ]


def cumac_scheduler_abstraction() -> CUMACSchedulerAbstraction:
    """Honest **cuMAC-style** (MAC/scheduling) narrative; not NVIDIA cuMAC."""
    return CUMACSchedulerAbstraction(
        scope="Abstract **cuMAC-style** scheduling story: map coexistence + fairness KPIs to resource-allocation narratives.",
        relation_to_twin="Gary scenario engine outputs (traffic, fairness, coverage pressure) are **policy inputs**, not MAC code.",
        execution="Real **cuMAC / MAC** runs only in **external** Aerial or lab RAN targets (`docs/EXTERNAL_RUNTIME_GAPS.md`).",
    )

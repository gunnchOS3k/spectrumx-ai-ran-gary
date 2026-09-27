"""Planning context separate from near-RT Gate 2 policy code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanningContext:
    site_id: str
    campus_slug: str
    phase: str
    geometry_fidelity: str
    source_manifest_sha256: str
    n_zones: int
    spectrum_mhz: float
    path_loss_exponent: float
    blockage_db: float
    design: dict[str, Any]
    twin: dict[str, Any]


def context_from_bundles(design: dict[str, Any], twin: dict[str, Any]) -> PlanningContext:
    return PlanningContext(
        site_id=str(design["site_id"]),
        campus_slug=str(design["campus_slug"]),
        phase=str(design["phase"]),
        geometry_fidelity=str(design.get("geometry_fidelity", "AUTHORED_PLANNING_LAYOUT")),
        source_manifest_sha256=str(design["source_manifest_sha256"]),
        n_zones=len(design.get("zones") or []),
        spectrum_mhz=float((design.get("spectrum") or {}).get("values", {}).get("budget_mhz", 20.0)),
        path_loss_exponent=2.2,
        blockage_db=float((design.get("blockage") or {}).get("values", {}).get("extra_db", 8.0)),
        design=design,
        twin=twin,
    )

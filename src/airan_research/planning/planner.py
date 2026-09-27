"""Non-real-time campus planning optimizer. Separate from Gate 2 twin_policies."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .candidate_generator import generate_candidates
from .objective_model import REQUIRED_OBJECTIVES, score
from .pareto import mark_pareto
from .planning_context import PlanningContext, context_from_bundles
from .propagation import optional_backend_status


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan(
    design: dict[str, Any],
    twin: dict[str, Any],
    *,
    producer_commit: str = "0" * 40,
    readygary_used: bool = False,
) -> dict[str, Any]:
    ctx = context_from_bundles(design, twin)
    if ctx.geometry_fidelity == "AUTHORED_PLANNING_LAYOUT":
        # Propagate; do not promote to surveyed.
        pass
    alts = []
    for cand in generate_candidates(ctx.site_id):
        objectives = score(
            nodes=int(cand["node_count"]),
            power_dbm=float(cand["power_dbm"]),
            site_id=ctx.site_id,
            exponent=ctx.path_loss_exponent,
            blockage_db=ctx.blockage_db,
            edge=bool(cand["edge_compute"]),
        )
        missing = [k for k in REQUIRED_OBJECTIVES if k not in objectives]
        if missing:
            raise RuntimeError(f"planner missing objectives: {missing}")
        alts.append(
            {
                "alternative_id": f"{ctx.site_id}-{cand['label']}",
                "label": cand["label"],
                "planning_variables": {
                    **cand,
                    "path_loss_exponent": ctx.path_loss_exponent,
                    "band_profile": (design.get("spectrum") or {}).get("values", {}).get("band_profile"),
                    "backhaul": "ntn_fallback" if ctx.site_id == "graham_land" else "terrestrial",
                },
                "objectives": objectives,
                "pareto_member": False,
                "notes": "Does not mutate campus/building geometry.",
            }
        )
    mark_pareto(alts)
    selected = next(a for a in alts if a["label"] == "balanced")
    commit = producer_commit if len(producer_commit) == 40 else "0" * 40
    return {
        "schema_name": "gunnchos.campus_optimization_result",
        "schema_version": "1.0.0",
        "run_id": f"planning-ric-v2-{ctx.site_id}",
        "site_id": ctx.site_id,
        "campus_slug": ctx.campus_slug,
        "phase": ctx.phase,
        "input_design_hash": _sha(design),
        "input_twin_state_hash": _sha(twin),
        "geometry_fidelity": ctx.geometry_fidelity,
        "source_manifest_sha256": ctx.source_manifest_sha256,
        "objectives": selected["objectives"],
        "alternatives": alts,
        "selected_alternative_id": selected["alternative_id"],
        "constraint_violations": [],
        "uncertainty": {
            "overall": "high",
            "notes": "Synthetic planning; no OTA evidence. Optional backends unused.",
        },
        "evidence_class": "SIMULATED",
        "readygary_used": bool(readygary_used),
        "ntn_used": ctx.site_id == "graham_land",
        "producer": {"repository": "spectrumx-ai-ran-gary", "commit": commit},
        "notes": json.dumps({"optional_backends": optional_backend_status()}),
    }


def plan_from_context(ctx: PlanningContext, **kwargs: Any) -> dict[str, Any]:
    return plan(ctx.design, ctx.twin, **kwargs)

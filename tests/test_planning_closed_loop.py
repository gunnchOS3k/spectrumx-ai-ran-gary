from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airan_research.calibration.holdout import calibrate
from airan_research.digital_twin_adapter import adapt_campus_twin, adapt_site_summary
from airan_research.planning.objective_model import REQUIRED_OBJECTIVES
from airan_research.planning.planner import plan
from airan_research.planning.propagation import require_optional_backend
from airan_research.ric.adapters import ReadOnlyTelemetryAdapter, SrsRANAdapter


def _minimal_pair(site_id: str = "gary") -> tuple[dict, dict]:
    slug = "graham-land" if site_id == "graham_land" else site_id
    design = {
        "site_id": site_id,
        "campus_slug": slug,
        "phase": "FULL",
        "geometry_fidelity": "AUTHORED_PLANNING_LAYOUT",
        "source_manifest_sha256": "520cbba99541b1505ebba899a2f34bfa905eadd7af8f5b5e8fa050c884067be1",
        "zones": [{"campus_requirement_id": "X"}],
        "spectrum": {"values": {"budget_mhz": 20.0, "band_profile": "n78_planning"}},
        "blockage": {"values": {"extra_db": 8.0}},
    }
    twin = {"site_id": site_id, "n_users": 16, "jains_fairness": 0.7}
    return design, twin


def test_legacy_adapter_unchanged():
    out = adapt_site_summary({"site_id": "ghana", "n_users": 12, "jains_fairness": 0.4})
    assert out == {"site_id": "ghana", "n_users": 12, "fairness_stub": 0.4}


def test_planner_exposes_required_objectives_and_pareto():
    design, twin = _minimal_pair()
    result = plan(design, twin)
    for key in REQUIRED_OBJECTIVES:
        assert key in result["objectives"]
    assert len(result["alternatives"]) >= 2
    assert any(a["pareto_member"] for a in result["alternatives"])
    assert result["readygary_used"] is False
    assert result["geometry_fidelity"] == "AUTHORED_PLANNING_LAYOUT"


def test_seven_campus_planning():
    sites = ["gary", "ghana", "guyana", "geelong", "germany", "gaza", "graham_land"]
    results = []
    for site in sites:
        design, twin = _minimal_pair(site)
        rich = adapt_campus_twin(twin, design)
        assert rich["geometry_fidelity"] == "AUTHORED_PLANNING_LAYOUT"
        result = plan(design, twin)
        measured = {
            "coverage": result["objectives"]["coverage"] - 0.04,
            "capacity": result["objectives"]["capacity"] - 1.5,
            "latency": result["objectives"]["latency"] + 3.0,
            "jitter": result["objectives"]["jitter"] + 0.4,
            "packet_loss": result["objectives"]["packet_loss"] + 0.2,
        }
        cal = calibrate(result, measured)
        assert cal["new_model_version"] != cal["parent_model_version"]
        assert cal["holdout_validation"]["pass"] is True
        results.append(result)
    assert len(results) == 7


def test_read_only_ric_and_fail_closed_optional():
    assert ReadOnlyTelemetryAdapter().actuate({"dry_run": True})["applied"] is False
    try:
        require_optional_backend("sionna")
        raise AssertionError("optional backend should fail closed")
    except RuntimeError:
        pass
    try:
        SrsRANAdapter().actuate({})
        raise AssertionError("srsRAN should fail closed")
    except RuntimeError:
        pass

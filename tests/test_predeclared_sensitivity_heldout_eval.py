"""Predeclared sensitivity must run the existing evaluator on held-out seeds.

SYNTHETIC_SIM only. The fixture metrics below are sentinels for call recording.
They are not a measured accuracy, latency, or fairness result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from airan_research.experiments.digital_programme import (
    PROTOCOL_RELPATH,
    load_protocol,
    run_predeclared_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_EVAL_ARTIFACT = ROOT / "results" / "experiments" / "predeclared_sensitivity_heldout_real_eval.json"


def _proto() -> dict:
    return {
        "split": {
            "train_seeds": [0, 1, 2, 3, 4],
            "held_out_seeds": [100, 101, 102, 103, 104],
            "n_episodes_per_seed": 2,
            "n_slots_per_episode": 4,
        },
        "synthetic_parameters": {
            "fidelity_continuity_factor": {
                "degraded": {"value": 0.85, "provenance": "SYNTHETIC_ASSUMPTION"},
            },
            "sensitivity_predeclared": {
                "parameter": "fidelity_continuity_factor.degraded",
                "values": [0.75],
                "seeds": [0, 1],
                "n_episodes_per_seed": 1,
                "n_slots_per_episode": 4,
                "policies": ["fixed_target_fidelity", "adaptive_fidelity"],
                "note": "SYNTHETIC_SIM predeclared sensitivity; not a physical RAN measurement.",
            },
        },
    }


def test_held_out_evaluator_records_metric_separately_from_train(monkeypatch):
    captured: list[dict] = []

    def fake_run_family(local, *, family, seeds, policies=None, ablation="full"):
        captured.append(
            {
                "split_held_out": list(local["split"]["held_out_seeds"]),
                "split_train": list(local["split"]["train_seeds"]),
                "family": family,
                "seeds": list(seeds),
            }
        )
        if family == "held_out":
            return {
                "policies": {
                    "fixed_target_fidelity": {
                        "service_continuity_utility": {"n": 1, "mean": -0.25}
                    }
                },
                "evidence_class": "SYNTHETIC_SIM",
            }
        return {
            "policies": {
                "fixed_target_fidelity": {
                    "service_continuity_utility": {"n": 1, "mean": 0.5}
                },
                "adaptive_fidelity": {
                    "service_continuity_utility": {"n": 1, "mean": 0.4}
                },
            },
            "evidence_class": "SYNTHETIC_SIM",
        }

    monkeypatch.setattr(
        "airan_research.experiments.digital_programme.run_family",
        fake_run_family,
    )
    out = run_predeclared_sensitivity(_proto())
    families = [call["family"] for call in captured]
    assert families == ["in_distribution", "held_out"]
    assert captured[0]["seeds"] == [0, 1]
    assert captured[1]["seeds"] == [100, 101]
    assert set(captured[0]["seeds"]).isdisjoint(captured[1]["seeds"])
    assert captured[1]["split_held_out"] == [100, 101]
    assert out["held_out_evaluated"] is True
    assert out["held_out_metric"] == "service_continuity_utility"
    assert out["evidence_class"] == "SYNTHETIC_SIM"
    row = out["rows"][0]
    assert row["policies"]["fixed_target_fidelity"]["mean"] == 0.5
    assert "held_out_policies" not in row["policies"]
    held = row["held_out_policies"]
    assert held["fixed_target_fidelity"]["mean"] == -0.25
    assert held["adaptive_fidelity"] is None
    assert held["fixed_target_fidelity"] is not row["policies"]["fixed_target_fidelity"]


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)


def test_real_predeclared_sensitivity_evaluator_matches_recorded_run():
    """Run the unmocked evaluator on the frozen protocol.

    SYNTHETIC_SIM only. This is not a physical measurement or a manuscript claim.
    """
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    first = run_predeclared_sensitivity(proto)
    second = run_predeclared_sensitivity(proto)
    assert _canon(first) == _canon(second)
    assert first["train_seeds"] == [0]
    assert first["held_out_seeds"] == [100]
    assert set(first["train_seeds"]).isdisjoint(first["held_out_seeds"])
    assert first["evidence_class"] == "SYNTHETIC_SIM"
    assert first["held_out_evaluated"] is True
    recorded = json.loads(
        REAL_EVAL_ARTIFACT.read_text(encoding="utf-8"),
        parse_constant=lambda token: float(token),
    )
    assert recorded["label"] == "SYNTHETIC_SIM"
    assert recorded["physical_measurement"] is False
    assert recorded["manuscript_claim"] is False
    assert recorded["source_sha"] == "2d70380725177a5603e94903e7073dd94fc83b3e"
    assert recorded["config_name"] == "sensitivity_predeclared"
    assert recorded["train_seeds"] == [0]
    assert recorded["held_out_seeds"] == [100]
    assert recorded["deterministic_across_two_invocations"] is True
    fresh = _canon(first)
    assert fresh == _canon(recorded["evaluator_output"])
    assert hashlib.sha256(fresh.encode("utf-8")).hexdigest() == recorded["evaluator_output_sha256"]
    assert _canon(recorded["service_continuity_utility"]["train"][0]["service_continuity_utility"]) == _canon(
        first["rows"][0]["policies"]
    )
    assert _canon(recorded["service_continuity_utility"]["held_out"][0]["service_continuity_utility"]) == _canon(
        first["rows"][0]["held_out_policies"]
    )

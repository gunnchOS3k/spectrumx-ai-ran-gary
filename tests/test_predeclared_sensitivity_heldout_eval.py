"""Predeclared sensitivity must run the existing evaluator on held-out seeds.

SYNTHETIC_SIM only. The fixture metrics below are sentinels for call recording.
They are not a measured accuracy, latency, or fairness result.
"""
from __future__ import annotations

from airan_research.experiments.digital_programme import run_predeclared_sensitivity


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

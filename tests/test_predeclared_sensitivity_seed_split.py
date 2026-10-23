"""Predeclared sensitivity must not reuse training seeds as held-out.

SYNTHETIC_SIM only. This does not run a physical RAN measurement and does not
record a new accuracy, latency, or fairness number.
"""
from __future__ import annotations

from airan_research.experiments.digital_programme import run_predeclared_sensitivity


def _proto(seeds: list[int]) -> dict:
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
                "seeds": list(seeds),
                "n_episodes_per_seed": 1,
                "n_slots_per_episode": 4,
                "policies": ["fixed_target_fidelity", "adaptive_fidelity"],
                "note": "SYNTHETIC_SIM predeclared sensitivity; not a physical RAN measurement.",
            },
        },
    }


def test_predeclared_sensitivity_held_out_seeds_do_not_overlap_train(monkeypatch):
    captured: list[dict] = []

    def fake_run_family(local, *, family, seeds, policies=None, ablation="full"):
        captured.append(
            {
                "split": dict(local["split"]),
                "family": family,
                "seeds": list(seeds),
            }
        )
        return {"policies": {}, "evidence_class": "SYNTHETIC_SIM"}

    monkeypatch.setattr(
        "airan_research.experiments.digital_programme.run_family",
        fake_run_family,
    )
    out = run_predeclared_sensitivity(_proto([0, 1]))
    assert captured, "sensitivity path did not record a split"
    split = captured[0]["split"]
    train = list(split["train_seeds"])
    held = list(split["held_out_seeds"])
    assert train == [0, 1]
    assert held == [100, 101]
    assert train != held
    assert set(train).isdisjoint(held)
    assert captured[0]["family"] == "in_distribution"
    assert captured[0]["seeds"] == train
    assert out["evidence_class"] == "SYNTHETIC_SIM"
    assert out["held_out_evaluated"] is False
    assert out["train_seeds"] == train
    assert out["held_out_seeds"] == held
    single = run_predeclared_sensitivity(_proto([0]))
    assert single["train_seeds"] == [0]
    assert single["held_out_seeds"] == [100]
    assert set(single["train_seeds"]).isdisjoint(single["held_out_seeds"])

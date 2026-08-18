"""Paper II digital programme tests (tiny splits; SYNTHETIC_SIM)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from airan_research.experiments.digital_programme import (
    PROTOCOL_RELPATH,
    Action,
    Slot,
    generate_slot,
    load_protocol,
    mean_ci,
    policy_information_equivalent,
    policy_no_adaptation,
    policy_twin_informed,
    predict_metrics,
    run_family,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_proto() -> dict:
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    proto["split"] = {
        **proto["split"],
        "train_seeds": [0],
        "held_out_seeds": [100],
        "n_episodes_per_seed": 2,
        "n_slots_per_episode": 3,
    }
    return proto


def test_protocol_is_frozen():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    assert proto["frozen"] is True
    assert proto["experiment_id"] == "rq2_cross_layer_continuity"
    assert proto["evidence_class"] == "SYNTHETIC_SIM"
    assert proto["latency_class"] == "HOST_PROCESS_TIMING"
    assert "oracle" in proto["policies"]
    assert "information_equivalent" in proto["policies"]


def test_tiny_family_runs_and_oracle_not_worse_than_no_adaptation():
    proto = _tiny_proto()
    out = run_family(
        proto,
        family="in_distribution",
        seeds=[0],
        policies=["no_adaptation", "twin_informed", "information_equivalent", "oracle"],
    )
    na = out["policies"]["no_adaptation"]["service_continuity_utility"]["mean"]
    oracle = out["policies"]["oracle"]["service_continuity_utility"]["mean"]
    assert oracle >= na - 1e-9
    assert out["evidence_class"] == "SYNTHETIC_SIM"


def test_info_equiv_and_twin_use_same_slot_features():
    rng = np.random.default_rng(0)
    slot = generate_slot(rng, "in_distribution")
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    a = policy_twin_informed(slot, None, 0, proto)
    b = policy_information_equivalent(slot, None, 0, proto)
    assert a.rationale.startswith("twin_informed")
    assert b.rationale == "information_equivalent"
    feats = slot.feature_vector()
    assert set(feats) >= {"latency_ms", "terrestrial_outage", "energy_budget", "continuity_strict"}


def test_no_adaptation_never_switches_network_when_terrestrial_stays():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    slot = Slot(
        latency_ms=30,
        jitter_ms=2,
        packet_loss_pct=1.0,
        spectrum_budget=20,
        energy_budget=80,
        n_users=4,
        user_priorities=[2, 2, 2, 2],
        terrestrial_outage=False,
        available_networks=["terrestrial", "local_edge_wifi"],
        edge_capacity=1.0,
        cloud_capacity=1.0,
        local_capacity=0.5,
        mobility=0.1,
        blockage=0.1,
        continuity_class="degraded_ok",
    )
    a1 = policy_no_adaptation(slot, None, 0, proto)
    a2 = policy_no_adaptation(slot, a1, 0, proto)
    m = predict_metrics(slot, a2, a1, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert a1.network == a2.network == "terrestrial"
    assert m["n_network_switches"] == 0.0


def test_mean_ci_from_real_values():
    stats = mean_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["n"] == 5
    assert abs(stats["mean"] - 0.3) < 1e-12
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]

"""RQ2 fidelity adaptation + checkpoint/recover adversarial tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from airan_research.experiments.digital_programme import (
    PROTOCOL_RELPATH,
    RECOVER_ACTION_PENALTY,
    STALE_CHECKPOINT_AGE,
    Action,
    Slot,
    information_equivalence_audit,
    load_protocol,
    policy_adaptive_checkpoint,
    policy_adaptive_fidelity,
    policy_checkpoint_disabled,
    policy_fixed_target_fidelity,
    predict_metrics,
    run_family,
)


ROOT = Path(__file__).resolve().parents[1]


def _slot(**overrides) -> Slot:
    base = dict(
        latency_ms=50,
        jitter_ms=3,
        packet_loss_pct=2.0,
        spectrum_budget=20,
        energy_budget=80,
        n_users=4,
        user_priorities=[2, 2, 2, 2],
        terrestrial_outage=False,
        available_networks=["terrestrial", "local_edge_wifi", "offline_continuation"],
        edge_capacity=1.0,
        cloud_capacity=1.0,
        local_capacity=0.5,
        mobility=0.2,
        blockage=0.2,
        continuity_class="degraded_ok",
        checkpoint_available=True,
        checkpoint_age_slots=1,
        task_progress=0.7,
        recovery_budget=0.15,
    )
    base.update(overrides)
    return Slot(**base)


def test_protocol_lists_fidelity_and_checkpoint_policies():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    assert "adaptive_fidelity" in proto["policies"]
    assert "fixed_target_fidelity" in proto["policies"]
    assert "adaptive_checkpoint" in proto["policies"]
    assert "checkpoint_disabled" in proto["policies"]
    assert proto["fidelity_levels"] == ["target", "degraded", "minimum_useful"]


def test_fixed_fidelity_baseline_holds_target():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    slot = _slot(energy_budget=20, terrestrial_outage=True)
    a = policy_fixed_target_fidelity(slot, None, 0, proto)
    assert a.fidelity_level == "target"
    assert a.checkpoint_action == "none"
    assert a.recover_action == "none"


def test_adaptive_fidelity_vs_fixed_baseline():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    stressed = _slot(energy_budget=25, terrestrial_outage=True, latency_ms=120)
    fixed = policy_fixed_target_fidelity(stressed, None, 0, proto)
    adaptive = policy_adaptive_fidelity(stressed, None, 0, proto)
    assert fixed.fidelity_level == "target"
    assert adaptive.fidelity_level == "minimum_useful"
    m_fixed = predict_metrics(
        stressed, fixed, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True
    )
    m_adapt = predict_metrics(
        stressed, adaptive, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True
    )
    # Adaptive minimum_useful uses lower energy factor; both finite
    assert m_adapt["energy_use_j"] < m_fixed["energy_use_j"]
    assert m_adapt["fidelity_level_code"] < m_fixed["fidelity_level_code"]


def test_checkpoint_disabled_baseline():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    a = policy_checkpoint_disabled(_slot(), None, 0, proto)
    assert a.checkpoint_action == "none"
    assert a.recover_action == "none"


def test_recover_unavailable_checkpoint():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    slot = _slot(checkpoint_available=False, terrestrial_outage=True, blockage=0.9)
    # Force recover action
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "local_edge_wifi", "edge", "probe", "degraded", "none", "recover")
    m = predict_metrics(slot, action, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["recover_failed"] == 1.0
    assert m["degraded_continuation"] == 1.0


def test_stale_checkpoint_fails_recover():
    slot = _slot(checkpoint_available=True, checkpoint_age_slots=STALE_CHECKPOINT_AGE + 1)
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "probe", "degraded", "none", "recover")
    m = predict_metrics(slot, action, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["recover_failed"] == 1.0


def test_recovery_cost_exceeds_budget():
    slot = _slot(recovery_budget=RECOVER_ACTION_PENALTY - 0.01)
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "probe", "degraded", "none", "recover")
    m = predict_metrics(slot, action, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["recover_failed"] == 1.0


def test_network_path_change_during_recovery_adds_cost():
    prev = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "prev")
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "ntn_fallback", "edge", "probe", "degraded", "none", "recover")
    slot = _slot(available_networks=["terrestrial", "ntn_fallback"], checkpoint_available=True, checkpoint_age_slots=1)
    m = predict_metrics(slot, action, prev, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["recover_cost"] >= RECOVER_ACTION_PENALTY + 0.03
    assert m["n_network_switches"] == 1.0


def test_thrash_prevention_penalty():
    prev = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "prev", "degraded", "checkpoint", "none")
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "probe", "degraded", "none", "recover")
    m = predict_metrics(_slot(), action, prev, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["thrash_event"] == 1.0


def test_adaptive_checkpoint_avoids_simultaneous_ckpt_and_recover():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    slot = _slot(terrestrial_outage=True, blockage=0.7, task_progress=0.9, energy_budget=90)
    a = policy_adaptive_checkpoint(slot, None, 0, proto)
    if a.recover_action == "recover":
        assert a.checkpoint_action == "none"


def test_information_equivalence_audit_records_observation_set():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    audit = information_equivalence_audit(_slot(), proto)
    assert "latency_ms" in audit["observation_set"]
    assert "checkpoint_available" in audit["observation_set"]
    assert audit["oracle_privileged_future"] is False
    assert audit["hidden_state_used"] is False
    assert audit["evidence_class"] == "SYNTHETIC_SIM"


def test_tiny_family_with_new_policies_runs():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    proto["split"] = {
        **proto["split"],
        "train_seeds": [0],
        "held_out_seeds": [100],
        "n_episodes_per_seed": 1,
        "n_slots_per_episode": 2,
    }
    out = run_family(
        proto,
        family="in_distribution",
        seeds=[0],
        policies=["fixed_target_fidelity", "adaptive_fidelity", "checkpoint_disabled", "adaptive_checkpoint"],
    )
    assert "adaptive_fidelity" in out["policies"]
    assert out["evidence_class"] == "SYNTHETIC_SIM"

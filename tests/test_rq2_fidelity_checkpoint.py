"""RQ2 fidelity adaptation + checkpoint/recover adversarial tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from airan_research.experiments.digital_programme import (
    ALLOWED_OBSERVATION_FIELDS,
    PROTOCOL_RELPATH,
    RECOVER_ACTION_PENALTY,
    STALE_CHECKPOINT_AGE,
    T_CRIT_VERIFICATION_SOURCE,
    THRASH_WINDOW,
    Action,
    CheckpointRuntimeState,
    PolicyObservation,
    Slot,
    information_equivalence_audit,
    load_protocol,
    mean_ci,
    observation_from_slot,
    policy_adaptive_checkpoint,
    policy_adaptive_fidelity,
    policy_checkpoint_disabled,
    policy_fixed_target_fidelity,
    predict_metrics,
    resolve_synthetic_params,
    run_family,
    run_predeclared_sensitivity,
    t_crit_975,
    validate_policy_observation_contract,
)


ROOT = Path(__file__).resolve().parents[1]

SCIPY_T_PPF_975_REFERENCE = {
    1: 12.7062047362,
    3: 3.1824463053,
    11: 2.2009851601,
    19: 2.0930240544,
    29: 2.0452296421,
}


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


def test_t_crit_matches_scipy_reference_for_required_dfs():
    for df, expected in SCIPY_T_PPF_975_REFERENCE.items():
        assert abs(t_crit_975(df) - expected) < 1e-9
    assert abs(t_crit_975(1_000_000) - 1.95996398454) < 1e-4
    assert "SciPy" in T_CRIT_VERIFICATION_SOURCE


def test_mean_ci_rejects_non_95_and_n_lt_2():
    with pytest.raises(ValueError, match="0.95"):
        mean_ci([1.0, 2.0], level=0.9)
    one = mean_ci([1.0])
    assert one["n"] == 1
    assert one["ci_low"] != one["mean"] or one["ci_low"] != one["ci_high"]
    import math

    assert math.isnan(one["ci_low"]) and math.isnan(one["ci_high"])


def test_protocol_lists_fidelity_and_checkpoint_policies():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    assert "adaptive_fidelity" in proto["policies"]
    assert "fixed_target_fidelity" in proto["policies"]
    assert "adaptive_checkpoint" in proto["policies"]
    assert "checkpoint_disabled" in proto["policies"]
    assert proto["fidelity_levels"] == ["target", "degraded", "minimum_useful"]
    assert proto["checkpoint_recovery"]["thrash_window"] == THRASH_WINDOW
    assert "synthetic_parameters" in proto
    assert proto["synthetic_parameters"]["fidelity_continuity_factor"]["degraded"]["provenance"] == (
        "SYNTHETIC_ASSUMPTION"
    )


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
    params = resolve_synthetic_params(proto)
    m_fixed = predict_metrics(
        stressed, fixed, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True, params=params
    )
    m_adapt = predict_metrics(
        stressed,
        adaptive,
        None,
        network_switch_penalty=0.05,
        placement_switch_penalty=0.03,
        apply_switch=True,
        params=params,
    )
    assert m_adapt["energy_use_j"] < m_fixed["energy_use_j"]
    assert m_adapt["fidelity_level_code"] < m_fixed["fidelity_level_code"]


def test_checkpoint_disabled_baseline():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    a = policy_checkpoint_disabled(_slot(), None, 0, proto)
    assert a.checkpoint_action == "none"
    assert a.recover_action == "none"


def test_recover_without_checkpoint_fails():
    slot = _slot(checkpoint_available=False, terrestrial_outage=True, blockage=0.9)
    action = Action([5, 5, 5, 5], [1, 1, 1, 1], "local_edge_wifi", "edge", "probe", "degraded", "none", "recover")
    m = predict_metrics(slot, action, None, network_switch_penalty=0.05, placement_switch_penalty=0.03, apply_switch=True)
    assert m["recover_failed"] == 1.0
    assert m["degraded_continuation"] == 1.0


def test_checkpoint_then_recover_state_machine():
    state = CheckpointRuntimeState(exists=False, age_slots=0, progress=0.0)
    assert state.exists is False
    ckpt = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "c", "degraded", "checkpoint", "none")
    thrash = state.record_and_apply(ckpt, slot_idx=0, task_progress=0.8, thrash_window=THRASH_WINDOW)
    assert thrash is False
    assert state.exists is True
    assert state.age_slots == 0
    assert state.progress == 0.8
    state.advance_age()
    assert state.age_slots == 1
    viewed = state.overlay_slot(_slot(checkpoint_available=False, checkpoint_age_slots=99, task_progress=0.2))
    assert viewed.checkpoint_available is True
    assert viewed.checkpoint_age_slots == 1
    recover = Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "r", "degraded", "none", "recover")
    thrash2 = state.record_and_apply(recover, slot_idx=1, task_progress=0.2, thrash_window=THRASH_WINDOW)
    assert thrash2 is True  # checkpoint then recover within rolling window
    m = predict_metrics(
        viewed,
        recover,
        ckpt,
        network_switch_penalty=0.05,
        placement_switch_penalty=0.03,
        apply_switch=True,
        thrash_event=thrash2,
    )
    assert m["recover_failed"] == 0.0
    assert m["thrash_event"] == 1.0


def test_newer_checkpoint_supersedes_older():
    state = CheckpointRuntimeState(exists=True, age_slots=5, progress=0.3)
    state.record_and_apply(
        Action([5, 5, 5, 5], [1, 1, 1, 1], "terrestrial", "cloud", "c", "degraded", "checkpoint", "none"),
        slot_idx=7,
        task_progress=0.9,
        thrash_window=THRASH_WINDOW,
    )
    assert state.age_slots == 0
    assert state.progress == 0.9
    assert state.last_checkpoint_slot == 7


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


def test_rolling_thrash_window_enforced():
    state = CheckpointRuntimeState()
    assert THRASH_WINDOW >= 2
    state.record_and_apply(
        Action([5] * 4, [1] * 4, "terrestrial", "cloud", "c", "degraded", "checkpoint", "none"),
        slot_idx=0,
        task_progress=0.5,
        thrash_window=THRASH_WINDOW,
    )
    # Fill with none so checkpoint ages out of a larger window if window were only 1 —
    # with window=2, immediate recover still thrash:
    thrash = state.record_and_apply(
        Action([5] * 4, [1] * 4, "terrestrial", "cloud", "r", "degraded", "none", "recover"),
        slot_idx=1,
        task_progress=0.5,
        thrash_window=THRASH_WINDOW,
    )
    assert thrash is True
    # After another none, window=[recover, none] → no thrash
    thrash2 = state.record_and_apply(
        Action([5] * 4, [1] * 4, "terrestrial", "cloud", "n", "degraded", "none", "none"),
        slot_idx=2,
        task_progress=0.5,
        thrash_window=THRASH_WINDOW,
    )
    assert thrash2 is False


def test_adaptive_checkpoint_avoids_simultaneous_ckpt_and_recover():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    slot = _slot(terrestrial_outage=True, blockage=0.7, task_progress=0.9, energy_budget=90)
    a = policy_adaptive_checkpoint(slot, None, 0, proto)
    if a.recover_action == "recover":
        assert a.checkpoint_action == "none"


def test_information_equivalence_audit_computes_booleans():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    audit = information_equivalence_audit(_slot(), proto)
    assert "latency_ms" in audit["observation_set"]
    assert "checkpoint_available" in audit["observation_set"]
    assert audit["oracle_privileged_future"] is False
    assert audit["oracle_privileged_model_eval"] is True
    assert audit["hidden_state_used"] is False
    assert audit["adaptive_uses_only_observation_set"] is True
    assert audit["baseline_uses_only_observation_set"] is True
    assert audit["information_equivalence_pass"] is True
    assert audit["evidence_class"] == "SYNTHETIC_SIM"
    for name in ("adaptive_fidelity", "fixed_target_fidelity"):
        assert audit["policy_contracts"][name]["required_subseteq_allowed"] is True


def test_prohibited_observation_access_fails_audit():
    obs = observation_from_slot(_slot(), "adaptive_fidelity")
    with pytest.raises(PermissionError):
        obs.get("future_slot_latency")  # type: ignore[arg-type]
    # Bad policy declares a prohibited required field → contract fails
    bad = validate_policy_observation_contract("adaptive_fidelity")
    assert bad["required_subseteq_allowed"] is True
    # Simulate a malicious required set
    from airan_research.experiments import digital_programme as dp

    original = dp.POLICY_REQUIRED_FIELDS["adaptive_fidelity"]
    try:
        dp.POLICY_REQUIRED_FIELDS["adaptive_fidelity"] = frozenset({"future_hidden_state"})
        contract = validate_policy_observation_contract("adaptive_fidelity")
        assert contract["required_subseteq_allowed"] is False
        assert "future_hidden_state" in contract["violations"]
        audit = information_equivalence_audit(_slot(), load_protocol(ROOT / PROTOCOL_RELPATH))
        assert audit["adaptive_uses_only_observation_set"] is False
        assert audit["information_equivalence_pass"] is False
    finally:
        dp.POLICY_REQUIRED_FIELDS["adaptive_fidelity"] = original


def test_policy_observation_restricted_fields():
    obs = PolicyObservation(
        fields={"latency_ms": 1.0, "energy_budget": 2.0},
        policy_name="probe",
        allowed_fields=ALLOWED_OBSERVATION_FIELDS,
    )
    assert obs.get("latency_ms") == 1.0
    with pytest.raises(PermissionError):
        obs.get("model_oracle_metric_eval")


def test_predeclared_sensitivity_runs():
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    out = run_predeclared_sensitivity(proto)
    assert out["ran"] is True
    assert out["predeclared"] is True
    assert out["outcome_tuned"] is False
    assert len(out["rows"]) == 3


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
    assert out["n_seeds"] == 1

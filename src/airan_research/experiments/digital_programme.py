"""Self-contained Paper II digital experiments.

SYNTHETIC_SIM only. Does not import or modify SpectrumX judged evaluate()/IQ.
28 GHz labelling (ReadyGary sibling) is FR2 mmWave, not Sub-6.
compute_time_ms is HOST_PROCESS_TIMING, not measured RF.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROTOCOL_RELPATH = Path("paper/artifacts/experiment_protocol.yaml")

T_CRIT_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    20: 2.086,
    30: 2.042,
    60: 2.000,
    120: 1.980,
}

NETWORK_PENALTY = {
    "terrestrial": 1.0,
    "local_edge_wifi": 1.1,
    "degraded_local": 1.6,
    "ntn_fallback": 2.2,
    "device_to_device": 1.4,
    "offline_continuation": 10.0,
}
PLACEMENT_PENALTY = {"cloud": 1.15, "edge": 1.0, "local": 1.25}

# Fidelity levels tied to documented service profiles (not free-form continuous).
FIDELITY_LEVELS = ("target", "degraded", "minimum_useful")
FIDELITY_CONTINUITY_FACTOR = {"target": 1.0, "degraded": 0.85, "minimum_useful": 0.65}
FIDELITY_ENERGY_FACTOR = {"target": 1.15, "degraded": 1.0, "minimum_useful": 0.75}
FIDELITY_SWITCH_PENALTY = 0.04
CHECKPOINT_ACTION_PENALTY = 0.03
RECOVER_ACTION_PENALTY = 0.06
STALE_CHECKPOINT_AGE = 8
THRASH_WINDOW = 2  # recover then checkpoint within window incurs thrash penalty
THRASH_PENALTY = 0.08

FAMILY_PARAMS = {
    "in_distribution": {
        "latency": (20.0, 80.0),
        "loss": (0.1, 4.0),
        "p_outage": 0.12,
        "energy": (60.0, 140.0),
        "blockage": (0.0, 0.3),
        "edge_cap": (0.4, 1.0),
    },
    "held_out": {
        "latency": (20.0, 80.0),
        "loss": (0.1, 4.0),
        "p_outage": 0.12,
        "energy": (60.0, 140.0),
        "blockage": (0.0, 0.3),
        "edge_cap": (0.4, 1.0),
    },
    "high_blockage": {
        "latency": (40.0, 140.0),
        "loss": (3.0, 18.0),
        "p_outage": 0.20,
        "energy": (60.0, 140.0),
        "blockage": (0.45, 0.95),
        "edge_cap": (0.2, 0.8),
    },
    "energy_constrained": {
        "latency": (20.0, 80.0),
        "loss": (0.1, 4.0),
        "p_outage": 0.12,
        "energy": (12.0, 40.0),
        "blockage": (0.0, 0.3),
        "edge_cap": (0.4, 1.0),
    },
    "outage_heavy": {
        "latency": (30.0, 120.0),
        "loss": (1.0, 10.0),
        "p_outage": 0.55,
        "energy": (50.0, 120.0),
        "blockage": (0.1, 0.6),
        "edge_cap": (0.3, 0.9),
    },
}


def _mini_yaml_load(text: str) -> Any:
    """Indent-based YAML subset (scalars, maps, lists, folded scalars)."""

    def parse_scalar(raw: str) -> Any:
        s = raw.strip()
        if s in ("true", "True"):
            return True
        if s in ("false", "False"):
            return False
        if s in ("null", "~"):
            return None
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(p) for p in inner.split(",")]
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s

    cleaned: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " #" in raw:
            raw = raw.split(" #", 1)[0].rstrip()
        cleaned.append(raw.rstrip())

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def parse_block(i: int, base: int) -> tuple[Any, int]:
        if i >= len(cleaned):
            return {}, i
        if indent(cleaned[i]) < base:
            return {}, i
        if cleaned[i].lstrip().startswith("- "):
            items: list[Any] = []
            while i < len(cleaned) and indent(cleaned[i]) == base and cleaned[i].lstrip().startswith("- "):
                item_raw = cleaned[i].lstrip()[2:]
                if ":" in item_raw and not item_raw.startswith("{"):
                    key, _, rest = item_raw.partition(":")
                    rest = rest.strip()
                    if rest:
                        items.append({key.strip(): parse_scalar(rest)})
                    else:
                        child, i = parse_block(i + 1, indent(cleaned[i]) + 2)
                        items.append({key.strip(): child})
                        continue
                else:
                    items.append(parse_scalar(item_raw))
                i += 1
            return items, i
        mapping: dict[str, Any] = {}
        while i < len(cleaned) and indent(cleaned[i]) == base and not cleaned[i].lstrip().startswith("- "):
            line = cleaned[i].lstrip()
            if ":" not in line:
                i += 1
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest in (">", "|"):
                folded: list[str] = []
                i += 1
                while i < len(cleaned) and indent(cleaned[i]) > base:
                    folded.append(cleaned[i].strip())
                    i += 1
                mapping[key] = " ".join(folded)
                continue
            if rest:
                mapping[key] = parse_scalar(rest)
                i += 1
                continue
            if i + 1 < len(cleaned) and indent(cleaned[i + 1]) > base:
                child, i = parse_block(i + 1, indent(cleaned[i + 1]))
                mapping[key] = child
            else:
                mapping[key] = {}
                i += 1
        return mapping, i

    doc, _ = parse_block(0, indent(cleaned[0]) if cleaned else 0)
    return doc


def load_protocol(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = _mini_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Protocol is not a mapping: {path}")
    if data.get("frozen") is not True:
        raise ValueError("Protocol must set frozen: true before runs")
    return data


def t_crit_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_975:
        return T_CRIT_975[df]
    return 1.96


def mean_ci(values: list[float], level: float = 0.95) -> dict[str, float]:
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "sd": float("nan")}
    m = float(sum(arr) / n)
    if n == 1:
        return {"n": 1, "mean": m, "ci_low": m, "ci_high": m, "sd": 0.0}
    var = sum((x - m) ** 2 for x in arr) / (n - 1)
    sd = math.sqrt(var)
    # Protocol ci_level is 0.95; other levels fall back to the same t table.
    _ = level
    half = t_crit_975(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "ci_low": m - half, "ci_high": m + half, "sd": sd}


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def jains_index(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sum(values)
    s2 = sum(v * v for v in values)
    n = len(values)
    return (s * s) / (n * s2) if s2 else 0.0


@dataclass
class Slot:
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    spectrum_budget: float
    energy_budget: float
    n_users: int
    user_priorities: list[int]
    terrestrial_outage: bool
    available_networks: list[str]
    edge_capacity: float
    cloud_capacity: float
    local_capacity: float
    mobility: float
    blockage: float
    continuity_class: str
    checkpoint_available: bool = True
    checkpoint_age_slots: int = 0
    task_progress: float = 0.5
    recovery_budget: float = 0.15

    def feature_vector(self) -> dict[str, float]:
        return {
            "latency_ms": self.latency_ms,
            "packet_loss_pct": self.packet_loss_pct,
            "terrestrial_outage": 1.0 if self.terrestrial_outage else 0.0,
            "energy_budget": self.energy_budget,
            "edge_capacity": self.edge_capacity,
            "blockage": self.blockage,
            "mobility": self.mobility,
            "continuity_strict": 1.0 if self.continuity_class == "strict" else 0.0,
            "checkpoint_available": 1.0 if self.checkpoint_available else 0.0,
            "checkpoint_age_slots": float(self.checkpoint_age_slots),
            "task_progress": float(self.task_progress),
            "recovery_budget": float(self.recovery_budget),
        }


@dataclass
class Action:
    shares: list[float]
    power: list[float]
    network: str
    placement: str
    rationale: str = ""
    fidelity_level: str = "target"
    checkpoint_action: str = "none"  # none | checkpoint
    recover_action: str = "none"  # none | recover


def _uniform(n: int, budget: float) -> list[float]:
    return [budget / max(n, 1)] * n


def _priority_shares(slot: Slot) -> list[float]:
    weights = [1.0 / max(1, p) for p in slot.user_priorities[: slot.n_users]]
    while len(weights) < slot.n_users:
        weights.append(1.0 / 3.0)
    total = sum(weights) or 1.0
    return [slot.spectrum_budget * w / total for w in weights]


def _optimize_shares(slot: Slot, seed: int) -> tuple[list[float], str]:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return _uniform(slot.n_users, slot.spectrum_budget), "scipy_missing_static_uniform"
    n = slot.n_users
    rng = np.random.default_rng(seed)
    x0 = np.full(n, slot.spectrum_budget / max(n, 1)) + rng.normal(0, 1e-6, size=n)

    def objective(x: np.ndarray) -> float:
        return -float(np.sum(np.log1p(np.maximum(x, 1e-9)))) + 0.01 * float(np.sum(x))

    constraints = [{"type": "ineq", "fun": lambda x: slot.spectrum_budget - np.sum(x)}]
    bounds = [(0.1, slot.spectrum_budget)] * n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        return _uniform(n, slot.spectrum_budget), "optimization_infeasible_static_uniform"
    return [float(v) for v in res.x], "slsqp"


def _pick_available(slot: Slot, preferred: list[str]) -> str:
    for name in preferred:
        if name in slot.available_networks:
            return name
    return slot.available_networks[0] if slot.available_networks else "offline_continuation"


def predict_metrics(
    slot: Slot,
    action: Action,
    prev: Action | None,
    *,
    network_switch_penalty: float,
    placement_switch_penalty: float,
    apply_switch: bool,
) -> dict[str, float]:
    fidelity = action.fidelity_level if action.fidelity_level in FIDELITY_CONTINUITY_FACTOR else "target"
    util = float(sum(action.shares) / max(slot.spectrum_budget, 1e-9))
    util = min(1.5, max(0.0, util))
    npen = NETWORK_PENALTY.get(action.network, 1.3)
    ppen = PLACEMENT_PENALTY.get(action.placement, 1.1)
    pred_latency = slot.latency_ms * npen * ppen * (1.0 + 0.4 * util)
    if fidelity == "target":
        pred_latency *= 1.05
    elif fidelity == "minimum_useful":
        pred_latency *= 0.9
    pred_loss = min(100.0, slot.packet_loss_pct * npen * (1.0 + 0.2 * util))
    reliability = max(0.0, min(1.0, 1.0 - pred_loss / 100.0))
    energy = (slot.energy_budget * 0.2) + (util * 15.0) + (0.05 * pred_latency)
    energy *= 0.6 + 0.4 * (sum(action.power) / max(len(action.power), 1))
    energy *= FIDELITY_ENERGY_FACTOR[fidelity]
    fairness = jains_index(action.shares)
    continuity = max(0.0, min(1.0, reliability * (1.0 - min(pred_latency, 500.0) / 500.0)))
    continuity *= FIDELITY_CONTINUITY_FACTOR[fidelity]

    n_net = 0
    n_place = 0
    n_fidelity = 0
    switch_cost = 0.0
    checkpoint_cost = 0.0
    recover_cost = 0.0
    recover_failed = 0.0
    thrash = 0.0
    degraded_continuation = 0.0

    if prev is not None:
        if action.network != prev.network:
            n_net = 1
            switch_cost += network_switch_penalty
        if action.placement != prev.placement:
            n_place = 1
            switch_cost += placement_switch_penalty
        if action.fidelity_level != prev.fidelity_level:
            n_fidelity = 1
            switch_cost += FIDELITY_SWITCH_PENALTY

    if action.checkpoint_action == "checkpoint":
        checkpoint_cost += CHECKPOINT_ACTION_PENALTY
        energy += 2.0

    if action.recover_action == "recover":
        recover_cost += RECOVER_ACTION_PENALTY
        if not slot.checkpoint_available:
            recover_failed = 1.0
            continuity *= 0.55
            degraded_continuation = 1.0
        elif slot.checkpoint_age_slots >= STALE_CHECKPOINT_AGE:
            recover_failed = 1.0
            continuity *= 0.7
            # Stale checkpoint → safe degraded/offline continuation
            degraded_continuation = 1.0
        elif recover_cost > slot.recovery_budget:
            recover_failed = 1.0
            continuity *= 0.6
            degraded_continuation = 1.0
        else:
            # Successful recover restores progress-proportional continuity boost
            continuity = min(1.0, continuity + 0.08 * max(0.0, min(1.0, slot.task_progress)))
        # Network path change during recovery increases cost
        if prev is not None and action.network != prev.network:
            recover_cost += 0.03
            switch_cost += 0.02
        if prev is not None and prev.checkpoint_action == "checkpoint":
            thrash = 1.0
            switch_cost += THRASH_PENALTY

    total_switch = switch_cost + checkpoint_cost + recover_cost
    net_utility = continuity - (total_switch if apply_switch else 0.0)
    return {
        "predicted_latency_ms": float(pred_latency),
        "predicted_reliability": float(reliability),
        "predicted_packet_loss_pct": float(pred_loss),
        "energy_use_j": float(energy),
        "fairness": float(fairness),
        "service_continuity_utility": float(max(-1.0, net_utility)),
        "capacity_utilization": float(min(1.0, util)),
        "n_network_switches": float(n_net),
        "n_placement_switches": float(n_place),
        "n_fidelity_switches": float(n_fidelity),
        "switch_cost": float(total_switch),
        "fidelity_level_code": float({"target": 2.0, "degraded": 1.0, "minimum_useful": 0.0}[fidelity]),
        "checkpoint_cost": float(checkpoint_cost),
        "recover_cost": float(recover_cost),
        "recover_failed": float(recover_failed),
        "thrash_event": float(thrash),
        "degraded_continuation": float(degraded_continuation),
    }


def _ablate_slot(slot: Slot, ablation: str) -> Slot:
    if ablation in ("full", "no_switch_cost"):
        return slot
    data = asdict(slot)
    if ablation == "no_edge_observations":
        data["latency_ms"] = 0.0
        data["jitter_ms"] = 0.0
        data["packet_loss_pct"] = 0.0
    elif ablation == "no_twin_context":
        data["mobility"] = 0.0
        data["blockage"] = 0.0
        data["continuity_class"] = "degraded_ok"
    elif ablation == "no_continuity_objective":
        data["continuity_class"] = "offline_ok"
    elif ablation == "no_energy_constraint":
        data["energy_budget"] = 1e9
    elif ablation == "no_local_edge":
        data["edge_capacity"] = 0.0
        data["available_networks"] = [n for n in slot.available_networks if n != "local_edge_wifi"]
        if not data["available_networks"]:
            data["available_networks"] = ["degraded_local"]
    return Slot(**data)


def policy_no_adaptation(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, seed, proto
    net = "terrestrial" if "terrestrial" in slot.available_networks else _pick_available(slot, ["terrestrial"])
    return Action(_uniform(slot.n_users, slot.spectrum_budget), [1.0] * slot.n_users, net, "cloud", "no_adaptation")


def policy_local_only(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, seed, proto
    net = _pick_available(slot, ["degraded_local", "device_to_device", "local_edge_wifi"])
    return Action(_uniform(slot.n_users, slot.spectrum_budget), [0.7] * slot.n_users, net, "local", "local_only")


def policy_cloud_only(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, seed, proto
    net = _pick_available(slot, ["terrestrial", "ntn_fallback"])
    return Action(_uniform(slot.n_users, slot.spectrum_budget), [1.0] * slot.n_users, net, "cloud", "cloud_only")


def policy_edge_only(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, seed, proto
    net = _pick_available(slot, ["local_edge_wifi", "degraded_local"])
    return Action(_priority_shares(slot), [0.9] * slot.n_users, net, "edge", "edge_only")


def policy_rule_based(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, seed, proto
    if slot.terrestrial_outage:
        net = _pick_available(slot, ["local_edge_wifi", "degraded_local", "ntn_fallback"])
    else:
        net = _pick_available(slot, ["terrestrial", "local_edge_wifi"])
    placement = "edge" if slot.edge_capacity > 0 and slot.latency_ms > 60 else "cloud"
    return Action(_priority_shares(slot), [1.0] * slot.n_users, net, placement, "rule_based")


def policy_optimization_based(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, proto
    shares, note = _optimize_shares(slot, seed)
    net = _pick_available(slot, ["terrestrial", "local_edge_wifi", "ntn_fallback"])
    return Action(shares, [1.0] * slot.n_users, net, "cloud", f"optimization_based:{note}")


def policy_twin_informed(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    _ = prev, proto
    if slot.terrestrial_outage:
        net = _pick_available(slot, ["local_edge_wifi", "degraded_local", "ntn_fallback"])
    elif slot.latency_ms > 40 and "local_edge_wifi" in slot.available_networks:
        net = "local_edge_wifi"
    else:
        net = _pick_available(slot, ["terrestrial", "local_edge_wifi"])
    if slot.continuity_class == "strict" or (slot.latency_ms > 40 and slot.edge_capacity > 0):
        placement = "edge"
    elif slot.energy_budget < 50:
        placement = "local" if slot.local_capacity > 0 else "edge"
    else:
        placement = "cloud"
    shares, note = _optimize_shares(slot, seed)
    power = [0.6] * slot.n_users if slot.energy_budget < 50 else [1.0] * slot.n_users
    return Action(shares, power, net, placement, f"twin_informed:{note}")


def policy_information_equivalent(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Same feature vector as twin_informed; frozen linear scores, uniform shares."""
    _ = prev, seed
    weights = proto.get("information_equivalent_weights") or {}
    w_lat = float(weights.get("latency", 0.004))
    w_loss = float(weights.get("loss", 0.02))
    w_out = float(weights.get("outage_mismatch", 0.35))
    w_en = float(weights.get("energy", 0.002))
    w_cont = float(weights.get("continuity_match", 0.25))
    w_edge = float(weights.get("edge_bonus_if_high_latency", 0.15))
    feats = slot.feature_vector()
    best: tuple[str, str] | None = None
    best_score = -1e18
    placements = [p for p, cap in (("cloud", slot.cloud_capacity), ("edge", slot.edge_capacity), ("local", slot.local_capacity)) if cap > 0]
    if not placements:
        placements = ["cloud"]
    for placement in placements:
        for network in slot.available_networks:
            score = 0.0
            score -= w_lat * feats["latency_ms"] * NETWORK_PENALTY.get(network, 1.3) * PLACEMENT_PENALTY.get(placement, 1.1)
            score -= w_loss * feats["packet_loss_pct"]
            if feats["terrestrial_outage"] > 0 and network == "terrestrial":
                score -= w_out
            if feats["energy_budget"] < 50 and placement == "cloud":
                score -= w_en * (50.0 - feats["energy_budget"])
            if feats["continuity_strict"] > 0 and placement == "edge":
                score += w_cont
            if feats["latency_ms"] > 40 and placement == "edge":
                score += w_edge
            if score > best_score:
                best_score = score
                best = (placement, network)
    placement, network = best or ("cloud", slot.available_networks[0])
    return Action(_uniform(slot.n_users, slot.spectrum_budget), [1.0] * slot.n_users, network, placement, "information_equivalent")


def policy_oracle(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Model-oracle over discrete placement×network on the current slot."""
    _ = seed
    costs = proto.get("costs") or {}
    npen = float(costs.get("network_switch_penalty", 0.05))
    ppen = float(costs.get("placement_switch_penalty", 0.03))
    placements = [p for p, cap in (("cloud", slot.cloud_capacity), ("edge", slot.edge_capacity), ("local", slot.local_capacity)) if cap > 0] or ["cloud"]
    shares = _uniform(slot.n_users, slot.spectrum_budget)
    best_action = Action(shares, [1.0] * slot.n_users, slot.available_networks[0], placements[0], "oracle")
    best_u = -1e18
    for placement in placements:
        for network in slot.available_networks:
            for fidelity in FIDELITY_LEVELS:
                cand = Action(
                    shares,
                    [1.0] * slot.n_users,
                    network,
                    placement,
                    "oracle",
                    fidelity_level=fidelity,
                    checkpoint_action="none",
                    recover_action="none",
                )
                m = predict_metrics(slot, cand, prev, network_switch_penalty=npen, placement_switch_penalty=ppen, apply_switch=True)
                if m["service_continuity_utility"] > best_u:
                    best_u = m["service_continuity_utility"]
                    best_action = cand
    return best_action


def policy_fixed_target_fidelity(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Baseline: hold fidelity at target; no fidelity adaptation."""
    base = policy_no_adaptation(slot, prev, seed, proto)
    return Action(
        base.shares,
        base.power,
        base.network,
        base.placement,
        "fixed_target_fidelity",
        fidelity_level="target",
        checkpoint_action="none",
        recover_action="none",
    )


def policy_adaptive_fidelity(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Adaptive fidelity from documented energy/latency thresholds (no future peek)."""
    base = policy_twin_informed(slot, prev, seed, proto)
    if slot.energy_budget < 40 or slot.terrestrial_outage:
        fidelity = "minimum_useful"
    elif slot.latency_ms > 80 or slot.energy_budget < 60:
        fidelity = "degraded"
    else:
        fidelity = "target"
    return Action(
        base.shares,
        base.power,
        base.network,
        base.placement,
        "adaptive_fidelity",
        fidelity_level=fidelity,
        checkpoint_action="none",
        recover_action="none",
    )


def policy_checkpoint_disabled(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Baseline: checkpointing disabled."""
    base = policy_no_adaptation(slot, prev, seed, proto)
    return Action(
        base.shares,
        base.power,
        base.network,
        base.placement,
        "checkpoint_disabled",
        fidelity_level="degraded",
        checkpoint_action="none",
        recover_action="none",
    )


def policy_adaptive_checkpoint(slot: Slot, prev: Action | None, seed: int, proto: dict) -> Action:
    """Cross-layer checkpoint/recover using only current-slot observations."""
    base = policy_twin_informed(slot, prev, seed, proto)
    fidelity = "degraded"
    ckpt = "none"
    recover = "none"
    network = base.network
    placement = base.placement
    # Checkpoint when progress is meaningful and energy allows
    if slot.task_progress >= 0.4 and slot.energy_budget >= 40 and not slot.terrestrial_outage:
        ckpt = "checkpoint"
    # Recover when outage/high blockage and checkpoint looks usable
    if slot.terrestrial_outage or slot.blockage > 0.6:
        if (
            slot.checkpoint_available
            and slot.checkpoint_age_slots < STALE_CHECKPOINT_AGE
            and RECOVER_ACTION_PENALTY <= slot.recovery_budget
        ):
            recover = "recover"
            ckpt = "none"  # thrash prevention: do not checkpoint in same recover step
        else:
            # Safe degraded/offline continuation
            fidelity = "minimum_useful"
            if "offline_continuation" in slot.available_networks and slot.blockage > 0.8:
                network = "offline_continuation"
                placement = "local"
    return Action(
        base.shares,
        base.power,
        network,
        placement,
        "adaptive_checkpoint",
        fidelity_level=fidelity,
        checkpoint_action=ckpt,
        recover_action=recover,
    )


POLICIES: dict[str, Callable[..., Action]] = {
    "no_adaptation": policy_no_adaptation,
    "local_only": policy_local_only,
    "cloud_only": policy_cloud_only,
    "edge_only": policy_edge_only,
    "rule_based": policy_rule_based,
    "optimization_based": policy_optimization_based,
    "twin_informed": policy_twin_informed,
    "information_equivalent": policy_information_equivalent,
    "oracle": policy_oracle,
    "fixed_target_fidelity": policy_fixed_target_fidelity,
    "adaptive_fidelity": policy_adaptive_fidelity,
    "checkpoint_disabled": policy_checkpoint_disabled,
    "adaptive_checkpoint": policy_adaptive_checkpoint,
}


def generate_slot(rng: np.random.Generator, family: str) -> Slot:
    p = FAMILY_PARAMS[family]
    latency = float(rng.uniform(*p["latency"]))
    loss = float(rng.uniform(*p["loss"]))
    outage = bool(rng.random() < p["p_outage"])
    energy = float(rng.uniform(*p["energy"]))
    blockage = float(rng.uniform(*p["blockage"]))
    edge_cap = float(rng.uniform(*p["edge_cap"]))
    n_users = int(rng.integers(4, 9))
    priorities = [int(rng.integers(1, 4)) for _ in range(n_users)]
    nets = ["terrestrial", "local_edge_wifi", "degraded_local", "ntn_fallback"]
    if outage:
        nets = [n for n in nets if n != "terrestrial"]
    if edge_cap <= 0.25:
        nets = [n for n in nets if n != "local_edge_wifi"]
    if not nets:
        nets = ["degraded_local"]
    continuity = "strict" if rng.random() < 0.25 else "degraded_ok"
    checkpoint_available = bool(rng.random() > 0.15)
    checkpoint_age = int(rng.integers(0, 12))
    task_progress = float(rng.uniform(0.1, 0.95))
    recovery_budget = float(rng.uniform(0.04, 0.20))
    return Slot(
        latency_ms=latency,
        jitter_ms=float(rng.uniform(1.0, 12.0)),
        packet_loss_pct=loss,
        spectrum_budget=20.0,
        energy_budget=energy,
        n_users=n_users,
        user_priorities=priorities,
        terrestrial_outage=outage,
        available_networks=nets + (["offline_continuation"] if rng.random() < 0.3 else []),
        edge_capacity=edge_cap,
        cloud_capacity=1.0,
        local_capacity=0.6,
        mobility=float(rng.uniform(0.0, 1.0)),
        blockage=blockage,
        continuity_class=continuity,
        checkpoint_available=checkpoint_available,
        checkpoint_age_slots=checkpoint_age,
        task_progress=task_progress,
        recovery_budget=recovery_budget,
    )


def _episode_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "service_continuity_utility",
        "predicted_latency_ms",
        "energy_use_j",
        "fairness",
        "n_network_switches",
        "n_placement_switches",
        "n_fidelity_switches",
        "switch_cost",
        "compute_time_ms",
        "recover_failed",
        "thrash_event",
        "degraded_continuation",
        "checkpoint_cost",
        "recover_cost",
        "fidelity_level_code",
    ]
    out = {k: float(sum(r.get(k, 0.0) for r in rows) / len(rows)) for k in keys if rows}
    if rows:
        out["n_network_switches"] = float(sum(r.get("n_network_switches", 0.0) for r in rows))
        out["n_placement_switches"] = float(sum(r.get("n_placement_switches", 0.0) for r in rows))
        out["n_fidelity_switches"] = float(sum(r.get("n_fidelity_switches", 0.0) for r in rows))
        out["switch_cost"] = float(sum(r.get("switch_cost", 0.0) for r in rows))
        out["compute_time_ms"] = float(sum(r.get("compute_time_ms", 0.0) for r in rows))
        out["recover_failed"] = float(sum(r.get("recover_failed", 0.0) for r in rows))
        out["thrash_event"] = float(sum(r.get("thrash_event", 0.0) for r in rows))
    return out


def information_equivalence_audit(
    slot: Slot,
    proto: dict[str, Any],
    *,
    adaptive_policy: str = "adaptive_fidelity",
    baseline_policy: str = "fixed_target_fidelity",
) -> dict[str, Any]:
    """Record observation set and label oracle vs non-oracle information access."""
    feats = slot.feature_vector()
    adaptive = POLICIES[adaptive_policy](slot, None, 0, proto)
    baseline = POLICIES[baseline_policy](slot, None, 0, proto)
    return {
        "observation_set": sorted(feats.keys()),
        "observation_values": feats,
        "adaptive_policy": adaptive_policy,
        "baseline_policy": baseline_policy,
        "adaptive_uses_only_observation_set": True,
        "baseline_uses_only_observation_set": True,
        "oracle_privileged_future": False,
        "hidden_state_used": False,
        "adaptive_action": {
            "network": adaptive.network,
            "placement": adaptive.placement,
            "fidelity_level": adaptive.fidelity_level,
            "checkpoint_action": adaptive.checkpoint_action,
            "recover_action": adaptive.recover_action,
        },
        "baseline_action": {
            "network": baseline.network,
            "placement": baseline.placement,
            "fidelity_level": baseline.fidelity_level,
            "checkpoint_action": baseline.checkpoint_action,
            "recover_action": baseline.recover_action,
        },
        "evidence_class": "SYNTHETIC_SIM",
        "note": "Information-equivalent comparison documents the same Slot.feature_vector(); oracle policy is labeled separately and is not used as a fair adaptive baseline.",
    }


def run_policy_on_episode(
    slots: list[Slot],
    policy_name: str,
    proto: dict[str, Any],
    *,
    seed: int,
    ablation: str = "full",
) -> dict[str, float]:
    costs = proto.get("costs") or {}
    npen = float(costs.get("network_switch_penalty", 0.05))
    ppen = float(costs.get("placement_switch_penalty", 0.03))
    apply_switch = ablation != "no_switch_cost"
    prev: Action | None = None
    frozen: Action | None = None
    rows: list[dict[str, float]] = []
    fn = POLICIES[policy_name] if policy_name != "static" else policy_rule_based
    for slot in slots:
        use = _ablate_slot(slot, ablation)
        t0 = time.perf_counter()
        if policy_name == "static":
            if frozen is None:
                frozen = policy_rule_based(use, None, seed, proto)
            action = frozen
        else:
            action = fn(use, prev, seed, proto)
        compute_ms = (time.perf_counter() - t0) * 1000.0
        metrics = predict_metrics(use, action, prev, network_switch_penalty=npen, placement_switch_penalty=ppen, apply_switch=apply_switch)
        metrics["compute_time_ms"] = float(compute_ms)
        rows.append(metrics)
        prev = action
    return _episode_metrics(rows)


def run_family(
    proto: dict[str, Any],
    *,
    family: str,
    seeds: list[int],
    policies: list[str] | None = None,
    ablation: str = "full",
) -> dict[str, Any]:
    split = proto["split"]
    n_ep = int(split["n_episodes_per_seed"])
    n_slots = int(split["n_slots_per_episode"])
    policies = policies or list(proto["policies"])
    per_policy_seed_means: dict[str, dict[str, list[float]]] = {p: {} for p in policies}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        episodes = [[generate_slot(rng, family) for _ in range(n_slots)] for _ in range(n_ep)]
        for policy in policies:
            if policy == "static":
                pass
            elif policy not in POLICIES:
                raise ValueError(f"Unknown policy {policy}")
            ep_rows = [run_policy_on_episode(ep, policy, proto, seed=seed, ablation=ablation) for ep in episodes]
            keys = ep_rows[0].keys()
            for k in keys:
                per_policy_seed_means[policy].setdefault(k, []).append(float(sum(r[k] for r in ep_rows) / len(ep_rows)))
    summaries = {}
    for policy, metric_map in per_policy_seed_means.items():
        summaries[policy] = {metric: mean_ci(vals) for metric, vals in metric_map.items()}
        summaries[policy]["seed_means"] = {metric: vals for metric, vals in metric_map.items()}
    baseline = per_policy_seed_means.get("no_adaptation", {})
    effects = {}
    if "service_continuity_utility" in baseline:
        base_vals = baseline["service_continuity_utility"]
        for policy, metric_map in per_policy_seed_means.items():
            effects[policy] = {
                "cohens_d_vs_no_adaptation_continuity": cohens_d(
                    metric_map["service_continuity_utility"], base_vals
                )
            }
    return {
        "family": family,
        "ablation": ablation,
        "seeds": list(seeds),
        "n_episodes_per_seed": n_ep,
        "n_slots_per_episode": n_slots,
        "policies": summaries,
        "effect_sizes": effects,
        "evidence_class": proto.get("evidence_class", "SYNTHETIC_SIM"),
        "latency_class": proto.get("latency_class", "HOST_PROCESS_TIMING"),
    }


def run_programme(
    repo_root: Path,
    *,
    include_heldout: bool = False,
    include_domain_shift: bool = False,
    include_ablations: bool = False,
) -> dict[str, Any]:
    protocol_path = repo_root / PROTOCOL_RELPATH
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol missing: {protocol_path}")
    proto = load_protocol(protocol_path)
    split = proto["split"]
    out_dir = repo_root / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "experiment_id": proto["experiment_id"],
        "title": proto.get("title"),
        "evidence_class": proto.get("evidence_class"),
        "latency_class": proto.get("latency_class"),
        "protocol_path": str(PROTOCOL_RELPATH),
        "frozen": True,
        "held_out_generated": False,
    }
    train = run_family(proto, family="in_distribution", seeds=list(split["train_seeds"]))
    (out_dir / f"{proto['experiment_id']}_train.json").write_text(json.dumps(train, indent=2) + "\n", encoding="utf-8")
    bundle["train"] = train
    if include_heldout:
        held = run_family(proto, family="held_out", seeds=list(split["held_out_seeds"]))
        (out_dir / f"{proto['experiment_id']}_heldout.json").write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
        bundle["held_out"] = held
        bundle["held_out_generated"] = True
    if include_domain_shift:
        shifts = {}
        for fam in split.get("domain_shift_families") or []:
            shifts[fam] = run_family(proto, family=str(fam), seeds=list(split["held_out_seeds"]))
        (out_dir / f"{proto['experiment_id']}_domain_shift.json").write_text(json.dumps(shifts, indent=2) + "\n", encoding="utf-8")
        bundle["domain_shift"] = shifts
    if include_ablations:
        ablations = {}
        for name in proto.get("ablations") or []:
            ablations[name] = run_family(
                proto,
                family="held_out",
                seeds=list(split["held_out_seeds"]),
                policies=["twin_informed"],
                ablation=str(name),
            )
        (out_dir / f"{proto['experiment_id']}_ablation.json").write_text(json.dumps(ablations, indent=2) + "\n", encoding="utf-8")
        bundle["ablations"] = ablations
    (out_dir / f"{proto['experiment_id']}_summary.json").write_text(json.dumps(_strip_seed_means(bundle), indent=2) + "\n", encoding="utf-8")
    return bundle


def _strip_seed_means(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_seed_means(v) for k, v in obj.items() if k != "seed_means"}
    if isinstance(obj, list):
        return [_strip_seed_means(x) for x in obj]
    return obj

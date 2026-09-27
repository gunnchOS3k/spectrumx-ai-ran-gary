"""Multi-objective planning vector. No opaque AI score as the only result."""
from __future__ import annotations

from .propagation import coverage_from_path_loss, log_distance_path_loss

REQUIRED_OBJECTIVES = (
    "coverage",
    "capacity",
    "latency",
    "jitter",
    "packet_loss",
    "spectral_efficiency",
    "unmet_demand",
    "jains_fairness",
    "zone_service_gap",
    "energy",
    "service_continuity",
    "recovery_time",
    "edge_compute_utilization",
    "planning_cost_proxy",
    "installation_complexity_proxy",
    "constraint_violations",
    "uncertainty",
)


def score(
    *,
    nodes: int,
    power_dbm: float,
    site_id: str,
    exponent: float,
    blockage_db: float,
    edge: bool,
) -> dict[str, float]:
    pl = log_distance_path_loss(
        12.0 + nodes,
        exponent=exponent,
        blockage_db=blockage_db,
        material_db=5.0 if site_id != "gaza" else 9.0,
    )
    coverage = coverage_from_path_loss(pl)
    if site_id == "gaza":
        coverage *= 0.86
    if site_id == "graham_land":
        coverage *= 0.8
    capacity = 8.0 * nodes + power_dbm * 0.4
    latency = 14.0 + (8.0 if site_id == "graham_land" else 0.0) + (6.0 if site_id == "gaza" else 0.0)
    latency += max(0.0, 4 - nodes) * 3.0
    jitter = 2.0 + (1.5 if site_id == "gaza" else 0.0)
    packet_loss = max(0.1, 1.8 - nodes * 0.3)
    se = capacity / 20.0
    unmet = max(0.0, 1.0 - coverage * (0.7 + 0.08 * nodes))
    fairness = 1.0 / (1.0 + 0.05 * abs(nodes - 2))
    gap = max(0.0, 1.0 - coverage)
    energy = 10.0 * nodes + max(power_dbm, 0.0) * 0.5
    continuity = 0.55 + 0.12 * nodes + (0.15 if edge else 0.0)
    if site_id == "gaza":
        continuity = min(1.0, continuity + 0.1)
    recovery = 40.0 / max(nodes, 1)
    edge_util = 0.35 + (0.25 if edge else 0.0)
    return {
        "coverage": round(min(1.0, coverage), 4),
        "capacity": round(capacity, 4),
        "latency": round(latency, 4),
        "jitter": round(jitter, 4),
        "packet_loss": round(min(100.0, packet_loss), 4),
        "spectral_efficiency": round(se, 4),
        "unmet_demand": round(min(1.0, unmet), 4),
        "jains_fairness": round(min(1.0, fairness), 4),
        "zone_service_gap": round(min(1.0, gap), 4),
        "energy": round(energy, 4),
        "service_continuity": round(min(1.0, continuity), 4),
        "recovery_time": round(recovery, 4),
        "edge_compute_utilization": round(min(1.0, edge_util), 4),
        "planning_cost_proxy": round(12.0 * nodes + (8.0 if edge else 0.0), 4),
        "installation_complexity_proxy": round(4.0 * nodes, 4),
        "constraint_violations": 0,
        "uncertainty": 0.72,
    }

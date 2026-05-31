def spectrum_utilization(allocations: list[float], capacity_mhz: float) -> float:
    return sum(allocations) / capacity_mhz if capacity_mhz else 0.0


def report_bundle(allocations: list[float], capacity_mhz: float, energy_w: float, demands: list[float] | None = None) -> dict:
    from .fairness import jains_index
    from .energy import energy_score

    denied = 0
    if demands:
        per_user = capacity_mhz / max(len(allocations), 1)
        denied = sum(1 for d in demands if d > per_user * 0.5)
    outage_proxy = round(denied / max(len(allocations), 1), 4)

    return {
        "spectrum_utilization": round(spectrum_utilization(allocations, capacity_mhz), 4),
        "jains_fairness": round(jains_index(allocations), 4),
        "energy_score": round(energy_score(energy_w, len(allocations)), 4),
        "user_denial_proxy": outage_proxy,
    }

from .policy_interface import proportional_fair_policy

def jain_fairness(allocations: list[float]) -> float:
    if not allocations or sum(allocations) == 0:
        return 0.0
    s, s2, n = sum(allocations), sum(x * x for x in allocations), len(allocations)
    return (s * s) / (n * s2) if s2 else 0.0


def energy_per_bit(power_w: float, throughput_bps: float) -> float:
    return power_w / throughput_bps if throughput_bps > 0 else float("inf")

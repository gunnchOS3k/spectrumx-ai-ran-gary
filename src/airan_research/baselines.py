from .policy_interface import (
    CombinedFairnessEnergyPolicy,
    EnergyAwarePolicy,
    FairnessAwarePolicy,
    PolicyContext,
    UniformPolicy,
)

POLICY_REGISTRY = {
    "baseline_uniform": UniformPolicy(),
    "fairness_aware": FairnessAwarePolicy(),
    "energy_aware": EnergyAwarePolicy(),
    "combined_fairness_energy": CombinedFairnessEnergyPolicy(),
}


def run_baseline(name: str, ctx: PolicyContext) -> list[float]:
    if name == "uniform":
        return UniformPolicy().allocate(ctx)
    if name == "fairness":
        return FairnessAwarePolicy().allocate(ctx)
    policy = POLICY_REGISTRY.get(name)
    if policy is None:
        raise ValueError(f"unknown baseline/policy: {name}")
    return policy.allocate(ctx)


def list_ablation_policies() -> list[str]:
    return list(POLICY_REGISTRY.keys())

from .policy_interface import FairnessAwarePolicy, PolicyContext, UniformPolicy


def run_baseline(name: str, ctx: PolicyContext) -> list[float]:
    policy = UniformPolicy() if name == "uniform" else FairnessAwarePolicy()
    return policy.allocate(ctx)

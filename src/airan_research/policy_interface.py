from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PolicyContext:
    n_users: int
    spectrum_mhz: float
    energy_budget_w: float


class Policy(Protocol):
    def allocate(self, ctx: PolicyContext) -> list[float]: ...


@dataclass
class UniformPolicy:
    def allocate(self, ctx: PolicyContext) -> list[float]:
        share = ctx.spectrum_mhz / max(ctx.n_users, 1)
        return [share] * ctx.n_users


@dataclass
class FairnessAwarePolicy:
    """Toy policy: equal share with mild energy scaling (research stub)."""

    def allocate(self, ctx: PolicyContext) -> list[float]:
        base = ctx.spectrum_mhz / max(ctx.n_users, 1)
        scale = min(1.0, ctx.energy_budget_w / 10.0)
        return [base * scale] * ctx.n_users

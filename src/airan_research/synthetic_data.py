"""Synthetic Gary-like RAN features (no private IQ data)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticRanSnapshot:
    n_users: int
    n_cells: int
    resource_blocks_mhz: float
    user_demands_mbps: list[float]
    seed: int


def generate(seed: int = 42, n_users: int = 50, n_cells: int = 5) -> SyntheticRanSnapshot:
    rng = random.Random(seed)
    return SyntheticRanSnapshot(
        n_users=n_users,
        n_cells=n_cells,
        resource_blocks_mhz=100.0,
        user_demands_mbps=[rng.uniform(1, 40) for _ in range(n_users)],
        seed=seed,
    )

"""AI-RAN research extension (does not modify competition evaluate path)."""
from dataclasses import dataclass


@dataclass
class PolicyAction:
    beam_id: int
    power_dbm: float
    resource_blocks: int


def proportional_fair_policy(demands: list[float], total_rb: int = 100) -> list[int]:
    total = sum(demands) or 1.0
    return [max(1, int(total_rb * d / total)) for d in demands]


def load_gary_site_schema_example() -> dict:
    """7GC-compatible site schema example (inline, no cross-repo dependency)."""
    return {
        "site_id": "gary",
        "is_flagship": True,
        "spectrum": {"bands_ghz": [3.5, 28], "constraint": "spectrum_limited"},
        "metrics": {"track": ["fairness", "spectral_efficiency", "energy_efficiency"]},
    }

"""Pareto front over planning alternatives."""
from __future__ import annotations

from typing import Any

MAXIMIZE = {
    "coverage",
    "capacity",
    "spectral_efficiency",
    "jains_fairness",
    "service_continuity",
    "edge_compute_utilization",
}


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    better_or_equal = True
    strictly_better = False
    for key, av in a.items():
        bv = b[key]
        if key in MAXIMIZE:
            if av < bv:
                better_or_equal = False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
            if av < bv:
                strictly_better = True
    return better_or_equal and strictly_better


def mark_pareto(alternatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for alt in alternatives:
        alt["pareto_member"] = not any(
            dominates(other["objectives"], alt["objectives"])
            for other in alternatives
            if other["alternative_id"] != alt["alternative_id"]
        )
    return alternatives

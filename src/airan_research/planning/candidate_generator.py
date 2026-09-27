"""Bounded planning candidates. Never mutates campus/building geometry."""
from __future__ import annotations

from typing import Any


def generate_candidates(site_id: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "sparse",
            "node_count": 1,
            "power_dbm": 17.0,
            "edge_compute": False,
            "geometry_mutated": False,
        },
        {
            "label": "balanced",
            "node_count": 2,
            "power_dbm": 20.0,
            "edge_compute": True,
            "geometry_mutated": False,
        },
        {
            "label": "dense",
            "node_count": 3,
            "power_dbm": 23.0,
            "edge_compute": True,
            "geometry_mutated": False,
        },
    ]

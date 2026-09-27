"""Planning report helpers."""
from __future__ import annotations

from typing import Any


def summarize(result: dict[str, Any]) -> str:
    obj = result["objectives"]
    return (
        f"{result['site_id']}: coverage={obj['coverage']} "
        f"capacity={obj['capacity']} latency={obj['latency']} "
        f"fairness={obj['jains_fairness']} alternatives={len(result['alternatives'])}"
    )

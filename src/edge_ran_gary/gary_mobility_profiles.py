"""
Mobility and **dynamic life** placeholders for the Gary extension.

Loads ``configs/wireless_scene/mobility_flow_profiles.json`` for UI and docs.
**Not** a traffic microsimulator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_mobility_flow_profiles(repo_root: Path) -> Dict[str, Any]:
    p = (repo_root / "configs" / "wireless_scene" / "mobility_flow_profiles.json").resolve()
    if not p.is_file():
        return {
            "loaded": False,
            "path": str(p),
            "flows": [],
            "note": "Add mobility_flow_profiles.json to describe ingress, buses, library events, queues.",
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"loaded": False, "path": str(p), "flows": [], "error": str(e)}
    flows = raw.get("flows") if isinstance(raw.get("flows"), list) else []
    return {
        "loaded": True,
        "path": str(p),
        "version": raw.get("version"),
        "documentation": raw.get("documentation"),
        "flows": flows,
    }


def flows_for_site(flows: List[Dict[str, Any]], site_id: str) -> List[Dict[str, Any]]:
    return [f for f in flows if isinstance(f, dict) and f.get("site_id") == site_id]

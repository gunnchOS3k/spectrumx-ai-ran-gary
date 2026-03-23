"""
Gary anchor-site geometry for the Completed Research Extension (micro-twin map).

Provides **recognizable building-like footprints** (simplified from map-aligned outlines),
optional overrides from ``configs/wireless_scene/site_footprints.json``, and optional
**glTF/GLB** placement from ``configs/wireless_scene/site_models.json``.

This module does **not** ship official competition data. Footprints are **approximate**
for visualization — replace with surveyed GIS or OSM exports when available.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONFIG_DIR = Path("configs/wireless_scene")
FOOTPRINTS_JSON = "site_footprints.json"
MODELS_JSON = "site_models.json"
ASSETS_MODEL_DIR = Path("assets/models")


def _shoelace_area_deg(polygon: List[List[float]]) -> float:
    """Approximate planar area (deg²) — for relative sizing only."""
    if len(polygon) < 3:
        return 0.0
    ring = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    s = 0.0
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        s += ring[i][0] * ring[j][1]
        s -= ring[j][0] * ring[i][1]
    return abs(s) / 2.0


def _approx_m2_from_deg2(area_deg2: float, lat_deg: float) -> float:
    """Rough m² from degree² at latitude (small-area approx)."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return area_deg2 * m_per_deg_lat * m_per_deg_lon


# ---------------------------------------------------------------------------
# Built-in footprints (many vertices — not axis-aligned “cubes”)
# Coordinates are [lon, lat]. Simplified civic / campus outlines for recognition.
# Replace via site_footprints.json when you have authoritative GIS.
# ---------------------------------------------------------------------------

_BUILTIN_SITES: Dict[str, Dict[str, Any]] = {
    "city_hall": {
        "id": "city_hall",
        "name": "Gary City Hall",
        "short_label": "🏛 Gary City Hall",
        "building_type": "Municipal civic",
        "site_type": "civic_government",
        "role": "Civic command center",
        "community_function": "Public services, hearings, permits, and resident-facing city business.",
        "why_matters": "Services, hearings, and emergency coordination need dependable links when residents need government most.",
        "height_m": 58,
        "footprint_source_note": "Built-in simplified outline near Broadway / civic center (replace with GIS).",
        "accent_line": [44, 62, 120, 255],
        "roof_tint_hint": [200, 210, 225],
        "polygon": [
            [-87.33828, 41.58422],
            [-87.33790, 41.58424],
            [-87.33762, 41.58418],
            [-87.33758, 41.58402],
            [-87.33772, 41.58398],
            [-87.33776, 41.58388],
            [-87.33795, 41.58382],
            [-87.33812, 41.58380],
            [-87.33822, 41.58386],
            [-87.33832, 41.58402],
            [-87.33828, 41.58422],
        ],
        "risk_bias": 0.55,
        "users": ["Residents (services)", "Civic staff", "Visitors / filers"],
        "gnb_offset_lon": 0.00022,
        "gnb_offset_lat": 0.00012,
        "demand_base_radius_m": 210,
    },
    "public_library": {
        "id": "public_library",
        "name": "Gary Public Library & Cultural Center",
        "short_label": "📚 Library & Cultural Center",
        "building_type": "Public library & cultural venue",
        "site_type": "library_cultural",
        "role": "Learning & inclusion hub",
        "community_function": "Homework, job search, cultural programming, and public meeting space.",
        "why_matters": "Patrons rely on Wi‑Fi and future cellular for homework, job search, and digital literacy.",
        "height_m": 42,
        "footprint_source_note": "Built-in L-shaped public-building outline (replace with GIS).",
        "accent_line": [120, 52, 140, 255],
        "roof_tint_hint": [220, 200, 235],
        "polygon": [
            [-87.33460, 41.58472],
            [-87.33390, 41.58474],
            [-87.33320, 41.58470],
            [-87.33312, 41.58455],
            [-87.33318, 41.58438],
            [-87.33345, 41.58432],
            [-87.33350, 41.58418],
            [-87.33405, 41.58414],
            [-87.33455, 41.58418],
            [-87.33468, 41.58435],
            [-87.33462, 41.58455],
            [-87.33460, 41.58472],
        ],
        "risk_bias": 0.40,
        "users": ["Patrons", "Study-room users", "Public-access learners"],
        "gnb_offset_lon": -0.00018,
        "gnb_offset_lat": 0.00015,
        "demand_base_radius_m": 235,
    },
    "west_side_leadership": {
        "id": "west_side_leadership",
        "name": "West Side Leadership Academy",
        "short_label": "🎓 West Side Leadership Academy",
        "building_type": "K–12 school campus",
        "site_type": "education_campus",
        "role": "Education & workforce pipeline",
        "community_function": "Instruction, student services, athletics, and family engagement on campus.",
        "why_matters": "Students and teachers need consistent access for instruction, safety comms, and take-home equity.",
        "height_m": 38,
        "footprint_source_note": "Built-in campus envelope (single outline; replace with parcel GIS).",
        "accent_line": [22, 120, 90, 255],
        "roof_tint_hint": [190, 225, 210],
        "polygon": [
            [-87.34855, 41.58542],
            [-87.34795, 41.58544],
            [-87.34725, 41.58540],
            [-87.34705, 41.58528],
            [-87.34702, 41.58505],
            [-87.34718, 41.58488],
            [-87.34755, 41.58478],
            [-87.34805, 41.58475],
            [-87.34845, 41.58482],
            [-87.34858, 41.58505],
            [-87.34855, 41.58542],
        ],
        "risk_bias": 0.65,
        "users": ["Students", "Teachers", "Staff"],
        "gnb_offset_lon": 0.0002,
        "gnb_offset_lat": -0.00014,
        "demand_base_radius_m": 265,
    },
}


def _deep_merge_base(site_id: str, override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(_BUILTIN_SITES[site_id])
    for k, v in override.items():
        if k == "polygon" and isinstance(v, list):
            out["polygon"] = v
        elif k in ("height_m", "footprint_approx_m2", "risk_bias"):
            out[k] = v
        elif isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def load_site_footprints_override(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load optional JSON; return None if missing or invalid."""
    p = (repo_root / CONFIG_DIR / FOOTPRINTS_JSON).resolve()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "sites" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def load_site_models_config(repo_root: Path) -> Dict[str, Any]:
    p = (repo_root / CONFIG_DIR / MODELS_JSON).resolve()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_glb_path(repo_root: Path, rel: str) -> Path:
    return (repo_root / rel).resolve()


def resolve_per_site_model(repo_root: Path, site_id: str, models_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    If config + file exist, return dict with scenegraph Path, scale, orientation_deg, anchor_offset_lonlat.
    Otherwise None.
    """
    sites = models_doc.get("sites") if models_doc else None
    if not isinstance(sites, dict):
        return None
    spec = sites.get(site_id)
    if not isinstance(spec, dict):
        return None
    rel = spec.get("glb_relative") or spec.get("glb_path")
    if not rel or not isinstance(rel, str):
        return None
    glb = _resolve_glb_path(repo_root, rel)
    if not glb.is_file():
        return None
    scale = spec.get("scale", 1.0)
    if isinstance(scale, (int, float)):
        scale = [float(scale), float(scale), float(scale)]
    orient = spec.get("orientation_deg", [0, 0, 0])
    if not isinstance(orient, list) or len(orient) < 3:
        orient = [0.0, 0.0, 0.0]
    off = spec.get("anchor_offset_lonlat", [0.0, 0.0])
    if not isinstance(off, list) or len(off) < 2:
        off = [0.0, 0.0]
    return {
        "scenegraph_path": glb,
        "scale": scale,
        "orientation_deg": [float(orient[0]), float(orient[1]), float(orient[2])],
        "anchor_offset_lonlat": [float(off[0]), float(off[1])],
    }


def build_anchor_buildings(repo_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (buildings, meta) where buildings match Streamlit expectations (id, name, polygon, …).

    meta includes:
    - geometry_source: "builtin" | "config_footprints"
    - model_specs: per-site resolved model or None
    """
    repo_root = repo_root.resolve()
    fp_doc = load_site_footprints_override(repo_root)
    models_doc = load_site_models_config(repo_root)

    order = ["city_hall", "public_library", "west_side_leadership"]
    buildings: List[Dict[str, Any]] = []
    geometry_source = "builtin"

    if fp_doc and isinstance(fp_doc.get("sites"), dict) and len(fp_doc["sites"]) > 0:
        geometry_source = "config_footprints"

    for site_id in order:
        if site_id not in _BUILTIN_SITES:
            continue
        if fp_doc and isinstance(fp_doc.get("sites"), dict) and site_id in fp_doc["sites"]:
            merged = _deep_merge_base(site_id, fp_doc["sites"][site_id])
        else:
            merged = deepcopy(_BUILTIN_SITES[site_id])

        poly = merged["polygon"]
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        lat_c = sum(lats) / len(lats)
        area_deg2 = _shoelace_area_deg(poly)
        merged["footprint_approx_m2"] = int(round(_approx_m2_from_deg2(area_deg2, lat_c)))
        merged["map_label"] = merged.get("short_label", merged["name"])
        merged["render_mode"] = "footprint_extruded"
        merged["geometry_note"] = merged.get("footprint_source_note", "")

        mspec = resolve_per_site_model(repo_root, site_id, models_doc)
        if mspec:
            merged["render_mode"] = "scenegraph"
            merged["_model_spec"] = mspec

        buildings.append(merged)

    model_specs = {b["id"]: b.get("_model_spec") for b in buildings}
    meta = {
        "geometry_source": geometry_source,
        "model_specs": model_specs,
        "footprints_config_path": str(repo_root / CONFIG_DIR / FOOTPRINTS_JSON),
        "models_config_path": str(repo_root / CONFIG_DIR / MODELS_JSON),
    }
    return buildings, meta


def apply_risk_colors_to_buildings(buildings: List[Dict[str, Any]]) -> None:
    """Mutate fill_color and line_color in place (risk bands + site accent border)."""
    for b in buildings:
        accent = b.get("accent_line", [40, 40, 40, 220])
        rs = float(b.get("risk_score", 0.5))
        if rs < 0.50:
            b["fill_color"] = [46, 204, 113, 210]
        elif rs < 0.70:
            b["fill_color"] = [241, 196, 15, 210]
        else:
            b["fill_color"] = [231, 76, 60, 210]
        # Stronger outline for site recognition
        b["line_color"] = [int(accent[0]), int(accent[1]), int(accent[2]), min(255, int(accent[3]))]

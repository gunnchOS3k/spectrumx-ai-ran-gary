"""
Occupancy, device density, and foot-traffic **visualization helpers** for the Gary micro-twin.

Produces pydeck-friendly data structures tied to `SiteScenarioState` (scenario engine).
All people/device placements are **aggregated / representative** — not one mesh per person.

Optional JSON (repo root):
- configs/wireless_scene/occupancy_profiles.json
- configs/wireless_scene/device_profiles.json
- configs/wireless_scene/movement_profiles.json

If absent, built-in defaults apply. See docs/OCCUPANCY_VISUALIZATION_PLAN.md.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONFIG_REL = Path("configs/wireless_scene")
OCC_PROFILE = "occupancy_profiles.json"
DEV_PROFILE = "device_profiles.json"
MOV_PROFILE = "movement_profiles.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_visual_profiles(repo_root: Path) -> Dict[str, Any]:
    root = repo_root.resolve()
    return {
        "occupancy": _load_json(root / CONFIG_REL / OCC_PROFILE),
        "device": _load_json(root / CONFIG_REL / DEV_PROFILE),
        "movement": _load_json(root / CONFIG_REL / MOV_PROFILE),
    }


def _point_in_poly(lon: float, lat: float, poly: List[List[float]]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    ring = poly if poly[0] != poly[-1] else poly[:-1]
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-18) + xi:
            inside = not inside
        j = i
    return inside


def _sample_points_in_polygon(poly: List[List[float]], n_target: int, seed: int) -> List[Tuple[float, float]]:
    """Rejection sample inside polygon ring (deterministic)."""
    ring = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    minx, maxx = min(lons), max(lons)
    miny, maxy = min(lats), max(lats)
    rng = random.Random(seed)
    out: List[Tuple[float, float]] = []
    attempts = 0
    while len(out) < n_target and attempts < n_target * 80:
        attempts += 1
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        if _point_in_poly(lon, lat, poly):
            out.append((lon, lat))
    if not out:
        cx, cy = sum(lons) / len(lons), sum(lats) / len(lats)
        out = [(cx, cy)]
    return out


def _activity_mode(preset: str, time_context: str, event_high: bool) -> str:
    if event_high:
        return "event_surge"
    if preset == "after_hours":
        return "after_hours"
    if preset == "emergency_special":
        return "emergency_ops"
    if preset == "peak_day":
        if time_context == "School hours":
            return "class_change_peak"
        return "peak_general"
    if time_context == "After hours":
        return "after_hours"
    if time_context == "Weekend":
        return "weekend"
    return "normal_day"


def _interpolate(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _poly_vertex(poly: List[List[float]], pick: str) -> List[float]:
    ring = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    if pick == "west":
        i = int(lons.index(min(lons)))
        return [ring[i][0], ring[i][1]]
    if pick == "east":
        i = int(lons.index(max(lons)))
        return [ring[i][0], ring[i][1]]
    if pick == "north":
        i = int(lats.index(max(lats)))
        return [ring[i][0], ring[i][1]]
    if pick == "south":
        i = int(lats.index(min(lats)))
        return [ring[i][0], ring[i][1]]
    return [sum(lons) / len(lons), sum(lats) / len(lats)]


def _build_flow_paths(
    site_id: str,
    poly: List[List[float]],
    centroid: List[float],
    mode: str,
) -> List[Dict[str, Any]]:
    """Static polylines (PathLayer) — illustrative foot traffic, not GPS traces."""
    clon, clat = centroid[0], centroid[1]
    core = [clon, clat]
    paths: List[Dict[str, Any]] = []

    def add(path_pts: List[List[float]], label: str, tip: str, color: List[int], width: int = 4):
        paths.append(
            {
                "path": path_pts,
                "label": label,
                "tip": tip,
                "color": color,
                "width": width,
            }
        )

    west = _poly_vertex(poly, "west")
    east = _poly_vertex(poly, "east")
    north = _poly_vertex(poly, "north")
    south = _poly_vertex(poly, "south")

    if site_id == "west_side_leadership":
        # Ingress → campus core; second leg = circulation / class change
        p1 = _interpolate(west, [clon, clat], 0.35)
        p2 = _interpolate(west, [clon, clat], 0.7)
        core = [clon, clat]
        wing = _interpolate(core, east, 0.55)
        if mode in ("class_change_peak", "event_surge"):
            add(
                [west, p1, p2, core, wing],
                "School flow (peak)",
                "**Aggregated** ingress + hallway-style movement during **peak / class-change** scenario (proxy path).",
                [41, 128, 185, 220],
                6,
            )
        elif mode == "after_hours":
            add(
                [south, _interpolate(south, core, 0.5), core],
                "After-hours presence",
                "Reduced **staff / activity** paths — scenario **after hours**.",
                [127, 140, 141, 160],
                3,
            )
        else:
            add(
                [west, p1, core],
                "School-day ingress",
                "**Representative** student/staff arrival pattern (not individual tracks).",
                [52, 152, 219, 200],
                4,
            )
            add(
                [core, wing],
                "Interior circulation",
                "Proxy **hallway / wing** movement tied to on-site population.",
                [46, 204, 113, 180],
                3,
            )

    elif site_id == "city_hall":
        entry = south
        svc = _interpolate(south, north, 0.45)
        queue_pt = _interpolate(entry, core, 0.55)
        add(
            [entry, queue_pt, svc],
            "Visitor / service queue",
            "**Visitor + staff** service movement (aggregated); workers vs public **zones** implied by path shape.",
            [142, 68, 173, 210],
            5 if mode in ("peak_general", "event_surge") else 4,
        )
        staff_side = _interpolate(east, core, 0.35)
        add(
            [east, staff_side, core],
            "Staff / back-office flow",
            "Municipal **staff** circulation proxy (separate entry emphasis).",
            [52, 73, 94, 170],
            3,
        )

    elif site_id == "public_library":
        entry = north
        reading = _interpolate(core, south, 0.4)
        event_rm = _interpolate(core, east, 0.35)
        add(
            [entry, _interpolate(entry, core, 0.5), core],
            "Patron entry",
            "Library **entry → lobby** flow (aggregated patrons).",
            [155, 89, 182, 200],
            4,
        )
        add(
            [core, reading],
            "Reading / study zone",
            "**Quiet study** area traffic proxy.",
            [241, 196, 15, 190],
            3,
        )
        if mode in ("event_surge", "peak_general", "weekend"):
            add(
                [core, event_rm],
                "Program / event room",
                "Higher **event / program** room use in peak or weekend scenarios.",
                [231, 76, 60, 185],
                4,
            )

    return paths


def build_pydeck_occupancy_bundle(
    buildings: List[Dict[str, Any]],
    preset_key: str,
    time_context: str,
    event_high: bool,
    repo_root: Path,
) -> Dict[str, Any]:
    """
    Returns dict with lists for HeatmapLayer, ScatterplotLayer clusters, PathLayer flows, TextLayer rows.
    All tied to each building's `_scenario` (SiteScenarioState fields).
    """
    profiles = load_visual_profiles(repo_root)
    occ_p = profiles.get("occupancy") or {}
    dev_p = profiles.get("device") or {}
    mov_p = profiles.get("movement") or {}

    n_heat_people = int(occ_p.get("heatmap_people_samples_per_site", 22))
    n_heat_dev = int(dev_p.get("heatmap_device_samples_per_site", 18))
    mode = str(mov_p.get("force_activity_mode") or _activity_mode(preset_key, time_context, event_high))

    heat_people: List[Dict[str, Any]] = []
    heat_devices: List[Dict[str, Any]] = []
    people_clusters: List[Dict[str, Any]] = []
    device_clusters: List[Dict[str, Any]] = []
    hero_markers: List[Dict[str, Any]] = []
    flow_paths: List[Dict[str, Any]] = []
    summary_labels: List[Dict[str, Any]] = []

    for b in buildings:
        st = b.get("_scenario")
        if st is None:
            continue
        poly = b["polygon"]
        sid = str(b["id"])
        clon, clat = b["centroid"][0], b["centroid"][1]
        seed = 17 + hash(sid) % 10000

        pts_p = _sample_points_in_polygon(poly, n_heat_people, seed)
        people = max(1.0, float(st.people_count))
        for i, (lon, lat) in enumerate(pts_p):
            # Slight variation so heat isn't flat
            bump = 0.75 + 0.5 * math.sin(i * 1.7 + people * 0.01)
            heat_people.append({"position": [lon, lat], "weight": (people / len(pts_p)) * bump})

        active_dev = max(1.0, float(st.active_ip_devices) + 1.5 * float(st.active_control_devices))
        pts_d = _sample_points_in_polygon(poly, n_heat_dev, seed + 3)
        for i, (lon, lat) in enumerate(pts_d):
            bump = 0.8 + 0.4 * math.cos(i + active_dev * 0.02)
            heat_devices.append({"position": [lon, lat], "weight": (active_dev / len(pts_d)) * bump})

        # Aggregated cluster disks (hero visualization)
        if sid == "west_side_leadership":
            stud = max(0.0, float(st.students_present))
            stf = max(0.0, float(st.staff_present))
            people_clusters.append(
                {
                    "position": [clon - 0.00011, clat + 0.00004],
                    "radius": min(220, 48 + stud * 0.045),
                    "label": "Student presence (aggregate)",
                    "tip": f"**~{stud:.0f} students** on campus (scenario) — disk size ∝ count · **not** individual people.",
                    "fill_color": [52, 152, 219, 95],
                    "line_color": [30, 90, 150, 200],
                }
            )
            people_clusters.append(
                {
                    "position": [clon + 0.00009, clat - 0.00007],
                    "radius": min(160, 40 + stf * 1.2),
                    "label": "Staff / teachers (aggregate)",
                    "tip": f"**~{stf:.0f} staff** (scenario).",
                    "fill_color": [46, 204, 113, 90],
                    "line_color": [20, 120, 80, 200],
                }
            )
        elif sid == "city_hall":
            wk = max(0.0, float(st.staff_present))
            vis = max(0.0, float(st.visitors_present))
            people_clusters.append(
                {
                    "position": [clon + 0.00006, clat + 0.00005],
                    "radius": min(200, 42 + wk * 0.85),
                    "label": "Workers on-site (aggregate)",
                    "tip": f"**~{wk:.0f} municipal workers** (scenario).",
                    "fill_color": [52, 73, 94, 100],
                    "line_color": [30, 50, 70, 220],
                }
            )
            people_clusters.append(
                {
                    "position": [clon - 0.0001, clat - 0.00006],
                    "radius": min(190, 38 + vis * 0.9),
                    "label": "Visitors / services (aggregate)",
                    "tip": f"**~{vis:.0f} visitor sessions** (scenario).",
                    "fill_color": [155, 89, 182, 95],
                    "line_color": [90, 50, 120, 210],
                }
            )
        else:  # library
            pat = max(0.0, float(st.people_count))
            people_clusters.append(
                {
                    "position": [clon - 0.00005, clat + 0.00006],
                    "radius": min(210, 44 + pat * 0.35),
                    "label": "Patrons / study load (aggregate)",
                    "tip": f"**~{pat:.0f} patrons** in building program scenario.",
                    "fill_color": [241, 196, 15, 100],
                    "line_color": [180, 130, 10, 220],
                }
            )
            people_clusters.append(
                {
                    "position": [clon + 0.0001, clat - 0.00005],
                    "radius": min(140, 36 + pat * 0.2),
                    "label": "Program / stacks zone",
                    "tip": "Secondary **patron density** zone (proxy split).",
                    "fill_color": [230, 126, 34, 85],
                    "line_color": [160, 80, 20, 200],
                }
            )

        # Device load clusters (IP vs control emphasis)
        ip_a = float(st.active_ip_devices)
        ctrl_a = float(st.active_control_devices)
        device_clusters.append(
            {
                "position": [clon + 0.00002, clat + 0.00011],
                "radius": min(200, 40 + ip_a * 0.035),
                "label": "Active IP devices (aggregate)",
                "tip": f"**{ip_a:.0f}** active IP endpoints (Wi‑Fi / handset **proxy**).",
                "fill_color": [142, 68, 173, 85],
                "line_color": [80, 40, 100, 210],
            }
        )
        device_clusters.append(
            {
                "position": [clon - 0.00008, clat + 0.0001],
                "radius": min(120, 28 + ctrl_a * 2.5),
                "label": "Control / LMR devices",
                "tip": f"**{ctrl_a:.0f}** active **control** radios / walkies (scenario).",
                "fill_color": [192, 57, 43, 90],
                "line_color": [120, 30, 25, 220],
            }
        )

        # Single representative marker per site (avoid clutter vs population)
        hero_markers.append(
            {
                "position": [clon + 0.00002, clat + 0.00007],
                "radius": 20,
                "label": "Representative presence",
                "tip": "**One hero marker per site** — illustrates that people are modeled, not a headcount of dots.",
                "fill_color": [52, 73, 94, 235],
                "line_color": [255, 255, 255, 255],
            }
        )

        flow_paths.extend(_build_flow_paths(sid, poly, [clon, clat], mode))

        total_d = float(st.ip_device_count + st.control_device_count)
        active_sum = float(st.active_ip_devices + st.active_control_devices)
        summary_labels.append(
            {
                "position": [clon + 0.00028, clat + 0.00018],
                "text": f"~{int(people)} people | {int(active_sum)} active dev",
                "label": f"{b['name'][:28]} — aggregates",
                "tip": (
                    f"**Scenario engine:** people **{people:.0f}**, active devices **{active_sum:.0f}** "
                    f"(IP+control), provisioned **{total_d:.0f}**. **Aggregated** overlay."
                ),
            }
        )

    return {
        "heatmap_people": heat_people,
        "heatmap_devices": heat_devices,
        "people_clusters": people_clusters,
        "device_clusters": device_clusters,
        "hero_markers": hero_markers,
        "flow_paths": flow_paths,
        "summary_labels": summary_labels,
        "activity_mode": mode,
        "heatmap_people_radius": int(occ_p.get("heatmap_people_radius_pixels", 58)),
        "heatmap_device_radius": int(dev_p.get("heatmap_device_radius_pixels", 48)),
    }

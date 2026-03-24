"""Shared helpers for Gary-anchored analytic export scripts (no secrets)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_anchor_centroids(r: Path) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(r))
    from src.edge_ran_gary.gary_site_geometry import build_anchor_buildings

    buildings, _ = build_anchor_buildings(r)
    rows = []
    for b in buildings:
        poly = b["polygon"]
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        lon_c = sum(lons) / len(lons)
        lat_c = sum(lats) / len(lats)
        rows.append(
            {
                "id": b["id"],
                "name": b["name"],
                "lon": float(lon_c),
                "lat": float(lat_c),
            }
        )
    return rows


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dx = (lon2 - lon1) * 111_320 * math.cos(math.radians(lat1))
    dy = (lat2 - lat1) * 110_540
    return float(math.hypot(dx, dy))


def fspl_db(d_m: float, fc_ghz: float = 3.5) -> float:
    """Free-space path loss (approx.), distance in meters, carrier in GHz."""
    d_km = max(d_m, 1.0) / 1000.0
    f_mhz = fc_ghz * 1000.0
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44


def try_import_sionna() -> tuple[bool, str]:
    try:
        import sionna

        return True, str(getattr(sionna, "__version__", "unknown"))
    except Exception as e:
        return False, f"{type(e).__name__}"


def try_import_deepmimo_hint() -> tuple[bool, str]:
    """DeepMIMO is often MATLAB-first; detect optional Python tooling only."""
    for mod in ("deepmimo", "DeepMIMO"):
        try:
            __import__(mod)
            return True, mod
        except Exception:
            continue
    return False, "no_python_deepmimo_module"

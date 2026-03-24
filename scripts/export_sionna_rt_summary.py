#!/usr/bin/env python3
"""
Write Sionna-shaped summaries under data/sionna_rt/ for the Streamlit hooks.

Uses a **deterministic analytic** path-loss outline from Gary anchor centroids (FSPL-style).
Always sets ``export_provenance.simulation_grade=analytic_fallback`` so the UI shows
**Loaded (demo summary)** when these files live under ``data/sionna_rt/``.

Optional ``--require-sionna-import`` exits non-zero if ``import sionna`` fails (CI / env check).

For a true **Loaded (simulation export)**, run a real Sionna RT scene yourself and write summaries with
``export_provenance.simulation_grade=full_solver`` (see docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md).

Outputs:
  - propagation_summary.json
  - path_loss_summary.json
  - coverage_grid.geojson
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from _gary_export_common import (
    distance_m,
    fspl_db,
    load_anchor_centroids,
    repo_root,
    try_import_sionna,
)


def _coverage_geojson(sites: list[dict], pl_db: list[float]) -> dict:
    feats = []
    for s, pl in zip(sites, pl_db):
        d = 0.00025
        lon, lat = s["lon"], s["lat"]
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "site_id": s["id"],
                    "label": s["name"][:40],
                    "path_loss_db_proxy": round(pl, 2),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon - d, lat - d],
                            [lon + d, lat - d],
                            [lon + d, lat + d],
                            [lon - d, lat + d],
                            [lon - d, lat - d],
                        ]
                    ],
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def build_payload(sites: list[dict], *, sionna_ok: bool, sionna_ver: str) -> tuple[dict, dict]:
    tx = sites[0]
    pl_vals = []
    los_like = []
    for s in sites:
        d = distance_m(tx["lon"], tx["lat"], s["lon"], s["lat"])
        pl = fspl_db(d, 3.5) + 6.0 * math.sin(0.01 * d)
        pl_vals.append(pl)
        los_like.append(1.0 if d < 450 else 0.35)

    mean_pl = sum(pl_vals) / len(pl_vals)
    los_frac = sum(los_like) / len(los_like)

    prov = {
        "script": "scripts/export_sionna_rt_summary.py",
        "simulation_grade": "analytic_fallback",
        "engine": "analytic",
        "sionna_import_ok": sionna_ok,
        "sionna_version_reported": sionna_ver if sionna_ok else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    body = {
        "sionna_export_version": "export_script_0.2",
        "solver": "analytic_gary_anchor",
        "export_provenance": prov,
        "sionna": {
            "scenario_name": "gary_anchor_analytic",
            "frequency_ghz": 3.5,
            "mean_path_loss_db": round(mean_pl, 2),
            "median_path_loss_db": round(sorted(pl_vals)[len(pl_vals) // 2], 2),
            "los_fraction": round(los_frac, 3),
            "num_sites": len(sites),
            "num_receiver_points": len(sites),
            "blockage_model": "none_analytic",
            "materials_summary": "not_modeled",
            "generated_at": prov["generated_at_utc"],
        },
        "coverage": {
            "mean_sinr_db": round(max(-5.0, 25.0 - mean_pl / 4.0), 2),
            "cell_edge_sinr_db": round(max(-8.0, 15.0 - mean_pl / 5.0), 2),
            "fraction_above_threshold": round(min(0.99, los_frac * 0.9 + 0.05), 3),
        },
    }
    path_loss = {**body, "export_kind": "path_loss_summary"}
    return body, path_loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=repo_root())
    ap.add_argument(
        "--require-sionna-import",
        action="store_true",
        help="Exit 1 if Sionna is not importable (does not change simulation_grade).",
    )
    args = ap.parse_args()
    r = args.repo_root.resolve()
    out = r / "data" / "sionna_rt"
    out.mkdir(parents=True, exist_ok=True)

    sionna_ok, sionna_ver = try_import_sionna()
    if args.require_sionna_import and not sionna_ok:
        print("ERROR: --require-sionna-import set but `import sionna` failed.", file=sys.stderr)
        return 1

    sites = load_anchor_centroids(r)
    prop, pls = build_payload(sites, sionna_ok=sionna_ok, sionna_ver=sionna_ver)

    (out / "propagation_summary.json").write_text(json.dumps(prop, indent=2), encoding="utf-8")
    (out / "path_loss_summary.json").write_text(json.dumps(pls, indent=2), encoding="utf-8")
    gj = _coverage_geojson(
        sites,
        [
            fspl_db(distance_m(sites[0]["lon"], sites[0]["lat"], s["lon"], s["lat"]), 3.5)
            for s in sites
        ],
    )
    (out / "coverage_grid.geojson").write_text(json.dumps(gj, indent=2), encoding="utf-8")

    print(f"Wrote propagation_summary.json, path_loss_summary.json, coverage_grid.geojson under {out.relative_to(r)}")
    print("UI: expect **Loaded (demo summary)** (analytic_fallback). For simulation export, supply real solver exports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

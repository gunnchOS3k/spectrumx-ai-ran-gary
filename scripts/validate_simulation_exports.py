#!/usr/bin/env python3
"""
Validate on-disk DeepMIMO- and Sionna-shaped exports before relying on Streamlit hooks.

**Does not** run DeepMIMO, Sionna RT, or AODT. Read-only JSON/GeoJSON checks only.

Usage (from repo root):

    python3 scripts/validate_simulation_exports.py
    python3 scripts/validate_simulation_exports.py --deepmimo
    python3 scripts/validate_simulation_exports.py --sionna

Provenance terms: `docs/PROVENANCE_LEGEND.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _gary_export_common import repo_root


def _err(msgs: List[str], msg: str) -> List[str]:
    msgs.append(msg)
    return msgs


def validate_deepmimo(repo: Path) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    d = repo / "data" / "deepmimo"
    primary = d / "scenario_summary.json"
    if not primary.is_file():
        return False, _err(msgs, f"Missing {primary.relative_to(repo)} (run scripts/export_deepmimo_summary.py)")

    try:
        raw: Dict[str, Any] = json.loads(primary.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, _err(msgs, f"Cannot read JSON: {primary}: {e}")

    if not isinstance(raw, dict):
        return False, _err(msgs, "scenario_summary.json root must be an object")
    if "deepmimo" not in raw:
        ok = False
        _err(msgs, "Missing top-level key `deepmimo`")
    prov = raw.get("export_provenance")
    if not isinstance(prov, dict):
        ok = False
        _err(msgs, "Missing or invalid `export_provenance` object")
    else:
        sg = str(prov.get("simulation_grade") or "")
        allowed = frozenset(
            {"analytic_fallback", "full_solver", "synthetic_template", "export_script_template"}
        )
        if sg not in allowed:
            ok = False
            _err(msgs, f"Unexpected export_provenance.simulation_grade: {sg!r}")

    meta = d / "scenario_meta.json"
    if meta.is_file():
        try:
            json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            ok = False
            _err(msgs, f"scenario_meta.json invalid: {e}")
    else:
        msgs.append(f"Optional missing: {meta.relative_to(repo)}")

    return ok, msgs


def validate_sionna(repo: Path) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    d = repo / "data" / "sionna_rt"
    prop = d / "propagation_summary.json"
    pls = d / "path_loss_summary.json"
    geo = d / "coverage_grid.geojson"

    for _, p in (("propagation_summary.json", prop), ("path_loss_summary.json", pls)):
        if not p.is_file():
            return False, _err(msgs, f"Missing {p.relative_to(repo)} (run scripts/export_sionna_rt_summary.py)")

    try:
        body = json.loads(prop.read_text(encoding="utf-8"))
        pl_body = json.loads(pls.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, _err(msgs, f"Cannot read Sionna JSON: {e}")

    if not isinstance(body, dict) or "sionna" not in body:
        ok = False
        _err(msgs, "propagation_summary.json must contain `sionna` object")
    if not isinstance(pl_body, dict):
        ok = False
        _err(msgs, "path_loss_summary.json root must be an object")
    prov = body.get("export_provenance")
    if not isinstance(prov, dict):
        ok = False
        _err(msgs, "Missing export_provenance on propagation_summary.json")

    if geo.is_file():
        try:
            gj = json.loads(geo.read_text(encoding="utf-8"))
            if gj.get("type") != "FeatureCollection":
                ok = False
                _err(msgs, "coverage_grid.geojson must be a FeatureCollection")
        except (json.JSONDecodeError, OSError) as e:
            ok = False
            _err(msgs, f"coverage_grid.geojson invalid: {e}")
    else:
        msgs.append(f"Optional missing: {geo.relative_to(repo)}")

    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate simulation export JSON under data/")
    ap.add_argument("--repo-root", type=Path, default=repo_root())
    ap.add_argument("--deepmimo", action="store_true", help="Only validate DeepMIMO drop zone")
    ap.add_argument("--sionna", action="store_true", help="Only validate Sionna RT drop zone")
    args = ap.parse_args()
    r = args.repo_root.resolve()

    do_dm = args.deepmimo or not args.sionna
    do_sn = args.sionna or not args.deepmimo
    if args.deepmimo and args.sionna:
        do_dm = do_sn = True

    ok_all = True
    if do_dm:
        ok, lines = validate_deepmimo(r)
        ok_all = ok_all and ok
        print("=== DeepMIMO (data/deepmimo) ===")
        for line in lines:
            print(line)
        print("STATUS:", "OK" if ok else "FAIL")
        print()

    if do_sn:
        ok, lines = validate_sionna(r)
        ok_all = ok_all and ok
        print("=== Sionna RT (data/sionna_rt) ===")
        for line in lines:
            print(line)
        print("STATUS:", "OK" if ok else "FAIL")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

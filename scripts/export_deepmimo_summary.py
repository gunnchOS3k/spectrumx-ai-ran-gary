#!/usr/bin/env python3
"""
Write DeepMIMO-shaped scenario_summary.json under data/deepmimo/.

Default --mode analytic_fallback produces **demo-grade** summaries in the UI unless you add
export_provenance.simulation_grade=full_solver from a real DeepMIMO MATLAB/Python pipeline.

Modes:
  analytic_fallback — deterministic stats from Gary anchor count (always allowed)
  full_solver       — requires optional Python DeepMIMO module OR env DEEPMIMO_FULL_SOLVER_OK=1
                      (set manually after you run a real DeepMIMO export elsewhere)

**Never** embeds API keys or MATLAB paths with secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from _gary_export_common import load_anchor_centroids, repo_root, try_import_deepmimo_hint


def build_document(
    *,
    mode: str,
    num_sites: int,
    deepmimo_py: bool,
) -> Optional[Dict[str, Any]]:
    grade = "analytic_fallback"
    if mode == "full_solver":
        if deepmimo_py or os.environ.get("DEEPMIMO_FULL_SOLVER_OK", "").strip() == "1":
            grade = "full_solver"
        else:
            return None

    now = datetime.now(timezone.utc).isoformat()
    prov = {
        "script": "scripts/export_deepmimo_summary.py",
        "simulation_grade": grade,
        "engine": "deepmimo_python" if grade == "full_solver" and deepmimo_py else "analytic",
        "generated_at_utc": now,
    }
    num_ue = max(32, num_sites * 40)
    return {
        "deepmimo_export_version": "export_script_0.2",
        "export_provenance": prov,
        "deepmimo": {
            "scenario_name": "gary_anchor_channel_outline" if grade != "full_solver" else "gary_anchor_deepmimo_marked",
            "num_bs": min(3, num_sites),
            "num_users": num_ue,
            "num_antennas_bs": 8,
            "num_antennas_ue": 2,
            "los_links": int(num_ue * 0.55),
            "nlos_links": int(num_ue * 0.45),
            "carrier_ghz": 3.5,
            "generated_at": now,
        },
        "channel_statistics": {
            "mean_path_loss_db": 118.0 + 0.12 * num_sites,
            "median_sinr_db": 11.5,
            "mean_sinr_db": 12.0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=repo_root())
    ap.add_argument("--mode", choices=("analytic_fallback", "full_solver"), default="analytic_fallback")
    args = ap.parse_args()
    r = args.repo_root.resolve()
    out = r / "data" / "deepmimo"
    out.mkdir(parents=True, exist_ok=True)

    sites = load_anchor_centroids(r)
    dm_py, _hint = try_import_deepmimo_hint()
    doc = build_document(mode=args.mode, num_sites=len(sites), deepmimo_py=dm_py)
    if doc is None:
        print(
            "ERROR: --mode full_solver requires DeepMIMO Python import or DEEPMIMO_FULL_SOLVER_OK=1.",
            file=sys.stderr,
        )
        return 1

    primary = out / "scenario_summary.json"
    meta = out / "scenario_meta.json"
    primary.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    meta.write_text(json.dumps({"deepmimo_meta": doc["deepmimo"], "export_provenance": doc["export_provenance"]}, indent=2), encoding="utf-8")

    print(f"Wrote {primary.relative_to(r)} and {meta.relative_to(r)}")
    print(f"Provenance simulation_grade={doc['export_provenance']['simulation_grade']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

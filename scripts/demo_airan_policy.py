#!/usr/bin/env python3
"""Toy AI-RAN research demo (synthetic; does not touch competition evaluate path)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airan_research.baselines import run_baseline
from airan_research.metrics import report_bundle
from airan_research.policy_interface import FairnessAwarePolicy, PolicyContext
from airan_research.report import write_reports
from airan_research.synthetic_data import generate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Run synthetic toy scenario")
    args = parser.parse_args()
    if not args.toy:
        print("Use --toy for synthetic demo", file=sys.stderr)
        return 2

    snap = generate()
    ctx = PolicyContext(n_users=snap.n_users, spectrum_mhz=snap.resource_blocks_mhz, energy_budget_w=8.0)
    baseline = run_baseline("uniform", ctx)
    ai = FairnessAwarePolicy().allocate(ctx)
    baseline_report = report_bundle(baseline, ctx.spectrum_mhz, ctx.energy_budget_w, snap.user_demands_mbps)
    ai_report = report_bundle(ai, ctx.spectrum_mhz, ctx.energy_budget_w * 0.9, snap.user_demands_mbps)

    out = {
        "mode": "toy",
        "seed": snap.seed,
        "baseline": baseline_report,
        "ai_policy": ai_report,
        "note": "research prototype — not competition IQ data",
    }
    print(json.dumps(out, indent=2))
    paths = write_reports(out, ROOT)
    print(f"Wrote {paths[0]}, {paths[1]}, {paths[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

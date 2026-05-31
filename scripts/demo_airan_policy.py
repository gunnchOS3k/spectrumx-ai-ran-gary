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
from airan_research.policy_interface import FairnessAwarePolicy, PolicyContext, UniformPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Run synthetic toy scenario")
    args = parser.parse_args()
    if not args.toy:
        print("Use --toy for synthetic demo", file=sys.stderr)
        return 2

    ctx = PolicyContext(n_users=50, spectrum_mhz=100.0, energy_budget_w=8.0)
    baseline = run_baseline("uniform", ctx)
    ai = FairnessAwarePolicy().allocate(ctx)
    baseline_report = report_bundle(baseline, ctx.spectrum_mhz, ctx.energy_budget_w)
    ai_report = report_bundle(ai, ctx.spectrum_mhz, ctx.energy_budget_w * 0.9)

    out = {
        "mode": "toy",
        "baseline": baseline_report,
        "ai_policy": ai_report,
        "note": "research prototype — not competition IQ data",
    }
    print(json.dumps(out, indent=2))
    e2e = ROOT / "results" / "e2e"
    e2e.mkdir(parents=True, exist_ok=True)
    (e2e / "airan_policy_demo.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = ["# AI-RAN Toy Demo (research path)", "", "## Baseline"] + [
        f"- **{k}**: {v}" for k, v in out.get("baseline", {}).items()
    ] + ["", "## AI policy"] + [f"- **{k}**: {v}" for k, v in out.get("ai_policy", {}).items()]
    lines += ["", out.get("note", "")]
    (e2e / "airan_policy_demo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {e2e / 'airan_policy_demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

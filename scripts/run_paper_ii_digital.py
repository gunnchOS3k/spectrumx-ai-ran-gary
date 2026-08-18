#!/usr/bin/env python3
"""Run Paper II digital programme. Protocol file must already exist."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airan_research.experiments.digital_programme import PROTOCOL_RELPATH, run_programme


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", action="store_true", help="Generate confirmatory held-out split")
    parser.add_argument("--domain-shift", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--all", action="store_true", help="Train + held-out + domain-shift + ablations")
    args = parser.parse_args()
    proto = ROOT / PROTOCOL_RELPATH
    if not proto.is_file():
        print(f"Refusing to run: protocol missing at {proto}", file=sys.stderr)
        return 2
    include_all = bool(args.all)
    bundle = run_programme(
        ROOT,
        include_heldout=include_all or args.held_out,
        include_domain_shift=include_all or args.domain_shift,
        include_ablations=include_all or args.ablations,
    )
    print(json.dumps({
        "experiment_id": bundle.get("experiment_id"),
        "held_out_generated": bundle.get("held_out_generated"),
        "evidence_class": bundle.get("evidence_class"),
        "latency_class": bundle.get("latency_class"),
        "families": [k for k in ("train", "held_out", "domain_shift", "ablations") if k in bundle],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

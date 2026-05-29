#!/usr/bin/env python3
"""Toy AI-RAN policy demo — synthetic data only."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airan_research.policy_interface import proportional_fair_policy, load_gary_site_schema_example
from airan_research.fairness import fairness_report
from airan_research.energy import energy_report
from airan_research.digital_twin_adapter import export_site_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Run toy synthetic demo")
    args = parser.parse_args()
    demands = [10.0, 20.0, 5.0, 15.0]
    alloc = proportional_fair_policy(demands)
    out = {
        "site": load_gary_site_schema_example(),
        "allocations_rb": alloc,
        "fairness": fairness_report(alloc),
        "energy": energy_report(12.0, 5e6),
        "twin_export": export_site_summary(),
        "mode": "toy_synthetic",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/airan_policy_toy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare toy AI-RAN policy variants on synthetic data."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airan_research.baselines import list_ablation_policies, run_baseline
from airan_research.metrics import report_bundle
from airan_research.policy_interface import PolicyContext
from airan_research.synthetic_data import generate

DEFAULT_SEED = 42
TABLE_FIELDS = [
    "policy",
    "seed",
    "spectrum_utilization",
    "jains_fairness",
    "energy_score",
    "user_denial_proxy",
    "mean_latency_ms_proxy",
]


def evaluate_policy(policy: str, seed: int) -> dict:
    snap = generate(seed=seed)
    ctx = PolicyContext(
        n_users=snap.n_users,
        spectrum_mhz=snap.resource_blocks_mhz,
        energy_budget_w=8.0,
    )
    allocations = run_baseline(policy, ctx)
    energy_w = ctx.energy_budget_w
    if policy in ("energy_aware", "combined_fairness_energy"):
        energy_w *= 0.9
    metrics = report_bundle(allocations, ctx.spectrum_mhz, energy_w, snap.user_demands_mbps)
    metrics["mean_latency_ms_proxy"] = round(
        sum(max(0.0, d - (ctx.spectrum_mhz / max(ctx.n_users, 1))) for d in snap.user_demands_mbps)
        / max(len(snap.user_demands_mbps), 1),
        4,
    )
    return {"policy": policy, "seed": seed, **metrics}


def save_table(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    policies = [r["policy"] for r in rows]
    fairness = [r["jains_fairness"] for r in rows]
    energy = [r["energy_score"] for r in rows]
    x = range(len(policies))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], fairness, width=width, label="Jain fairness")
    ax.bar([i + width / 2 for i in x], energy, width=width, label="energy score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylabel("normalized score (toy)")
    ax.set_title(f"Ablation seed={rows[0]['seed']} (synthetic-only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = [evaluate_policy(name, args.seed) for name in list_ablation_policies()]
    table_path = ROOT / "results" / "ablation" / "ablation_table.csv"
    figure_path = ROOT / "figures" / "ablation" / "ablation_fairness_energy.png"
    save_table(rows, table_path)
    save_figure(rows, figure_path)
    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

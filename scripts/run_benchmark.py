#!/usr/bin/env python3
"""Deterministic synthetic AI-RAN benchmark (no private data)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airan_research.baselines import run_baseline
from airan_research.metrics import report_bundle
from airan_research.policy_interface import PolicyContext
from airan_research.synthetic_data import generate

DEFAULT_SEED = 42
METRIC_FIELDS = [
    "run_id",
    "seed",
    "policy",
    "spectrum_utilization",
    "jains_fairness",
    "energy_score",
    "user_denial_proxy",
    "mean_latency_ms_proxy",
]


def run_single(seed: int, policy: str = "fairness_aware") -> dict:
    snap = generate(seed=seed)
    ctx = PolicyContext(
        n_users=snap.n_users,
        spectrum_mhz=snap.resource_blocks_mhz,
        energy_budget_w=8.0,
    )
    allocations = run_baseline(policy, ctx)
    energy_w = ctx.energy_budget_w * (0.95 if policy != "baseline_uniform" else 1.0)
    metrics = report_bundle(allocations, ctx.spectrum_mhz, energy_w, snap.user_demands_mbps)
    metrics["mean_latency_ms_proxy"] = round(
        sum(max(0.0, d - (ctx.spectrum_mhz / max(ctx.n_users, 1))) for d in snap.user_demands_mbps)
        / max(len(snap.user_demands_mbps), 1),
        4,
    )
    return {
        "run_id": f"seed_{seed}",
        "seed": seed,
        "policy": policy,
        **metrics,
    }


def save_metrics(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    seeds = [r["seed"] for r in rows]
    util = [r["spectrum_utilization"] for r in rows]
    fairness = [r["jains_fairness"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(seeds, util, marker="o", label="spectrum_utilization")
    ax1.set_xlabel("seed")
    ax1.set_ylabel("spectrum utilization")
    ax2 = ax1.twinx()
    ax2.plot(seeds, fairness, marker="s", color="tab:orange", label="jains_fairness")
    ax2.set_ylabel("Jain fairness")
    fig.suptitle("Synthetic AI-RAN benchmark (research prototype)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--policy", default="fairness_aware")
    args = parser.parse_args()

    seeds = [args.seed] if args.seeds == [42, 43, 44, 45, 46] and args.seed != DEFAULT_SEED else args.seeds
    rows = [run_single(seed=s, policy=args.policy) for s in seeds]

    metrics_path = ROOT / "results" / "benchmark" / "metrics.csv"
    figure_path = ROOT / "figures" / "benchmark" / "benchmark_metrics.png"
    save_metrics(rows, metrics_path)
    save_plot(rows, figure_path)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

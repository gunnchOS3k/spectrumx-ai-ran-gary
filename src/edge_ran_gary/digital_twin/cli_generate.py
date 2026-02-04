"""
CLI to generate Gary Micro-Twin synthetic IQ dataset.

Usage:
    python -m edge_ran_gary.digital_twin.cli_generate --config configs/gary_micro_twin.yaml --n 50 --out data/synthetic/gary_micro_twin

Outputs:
    - <out>/*.npy (IQ files, gitignored)
    - <out>/metadata.csv (gitignored)
    - <out>/summary.json (counts per zone / signal_type / label)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .gary_micro_twin import GaryMicroTwin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gary Micro-Twin synthetic 1s IQ windows + metadata."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gary_micro_twin.yaml",
        help="Path to gary_micro_twin YAML config",
    )
    parser.add_argument(
        "-n", "--n",
        type=int,
        default=50,
        help="Total number of samples to generate (split across zones)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic/gary_micro_twin",
        help="Output directory for .npy, metadata.csv, summary.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--label-balance",
        type=float,
        default=0.5,
        help="Fraction of samples with label=1",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # n_per_zone so total ≈ n
    micro_twin = GaryMicroTwin(config_path=str(config_path))
    n_zones = len(micro_twin.zone_model.zone_ids)
    n_per_zone = max(1, args.n // n_zones)
    total = n_per_zone * n_zones

    samples, metadata_df = micro_twin.generate_samples_per_zone(
        n_per_zone=n_per_zone,
        label_balance=args.label_balance,
        seed=args.seed,
    )

    micro_twin.save_dataset(str(out_path), samples, metadata_df)

    # summary.json: counts per zone, signal_type, label
    summary = {
        "n_samples": total,
        "seed": args.seed,
        "config": str(config_path),
        "by_label": metadata_df["label"].value_counts().to_dict(),
        "by_zone": metadata_df["zone_id"].value_counts().to_dict(),
    }
    if "signal_type" in metadata_df.columns:
        summary["by_signal_type"] = metadata_df["signal_type"].value_counts().to_dict()
    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

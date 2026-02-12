"""
CLI: Generate Gary Micro-Twin synthetic dataset (competition-safe).

  python -m edge_ran_gary.digital_twin.generate --config configs/gary_micro_twin.yaml --n 200 --out data/synthetic/gary_micro_twin

Writes:
  - out/iq/*.npy
  - out/metadata.csv
  - out/manifest.json (resolved config + hashes + generator_version + timestamp)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .contracts import GENERATOR_VERSION
from .samples import generate_sample, _config_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gary Micro-Twin synthetic IQ dataset.")
    parser.add_argument("--config", type=str, default="configs/gary_micro_twin.yaml", help="Path to gary_micro_twin.yaml")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/synthetic/gary_micro_twin", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label-balance", type=float, default=0.5, help="Fraction of samples with label=1")
    args = parser.parse_args()

    out = Path(args.out)
    iq_dir = out / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    zone_ids = list(config.get("zones", {}).keys())
    if not zone_ids:
        raise ValueError("Config has no zones")

    n_per_zone = max(1, args.n // len(zone_ids))
    config_hash = _config_hash(args.config)
    rows = []
    sample_idx = 0

    for zid in zone_ids:
        n1 = int(n_per_zone * args.label_balance)
        n0 = n_per_zone - n1
        for _ in range(n0):
            s = generate_sample(seed=args.seed + sample_idx, zone_id=zid, label=0, config_path=args.config)
            fname = f"sample_{sample_idx:06d}.npy"
            (iq_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            np.save(iq_dir / fname, s.iq)
            rows.append({"file": f"iq/{fname}", "label": s.label, **s.metadata})
            sample_idx += 1
        for _ in range(n1):
            s = generate_sample(seed=args.seed + sample_idx, zone_id=zid, label=1, config_path=args.config)
            fname = f"sample_{sample_idx:06d}.npy"
            np.save(iq_dir / fname, s.iq)
            rows.append({"file": f"iq/{fname}", "label": s.label, **s.metadata})
            sample_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)

    manifest = {
        "config_path": str(Path(args.config).resolve()),
        "config_hash": config_hash,
        "generator_version": GENERATOR_VERSION,
        "n_samples": len(rows),
        "seed": args.seed,
        "label_balance": args.label_balance,
        "zone_ids": zone_ids,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(rows)} samples to {out}")
    print(f"  iq/*.npy, metadata.csv, manifest.json")


if __name__ == "__main__":
    main()

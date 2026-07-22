"""AI-RAN Gate 2 CLI (`airan`)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from airan_research.gate2.twin_policies import (
    POLICY_NAMES,
    evaluate_policy,
    run_ablations,
    run_benchmark,
    validate_decision,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airan")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate-policy")
    ev.add_argument("--twin-state", required=True)
    ev.add_argument("--policy", required=True, choices=list(POLICY_NAMES))
    ev.add_argument("--output", required=True)
    ev.add_argument("--schema-dir", default=None)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--readygary-json", default=None)

    vd = sub.add_parser("validate-decision")
    vd.add_argument("path")
    vd.add_argument("--schema-dir", default=None)

    bm = sub.add_parser("benchmark")
    bm.add_argument("--twin-state", required=True)
    bm.add_argument("--output", required=True)
    bm.add_argument("--schema-dir", default=None)
    bm.add_argument("--repetitions", type=int, default=5)
    bm.add_argument("--warmup", type=int, default=1)
    bm.add_argument("--seed", type=int, default=0)

    ab = sub.add_parser("ablation")
    ab.add_argument("--twin-state", required=True)
    ab.add_argument("--output", required=True)
    ab.add_argument("--schema-dir", default=None)
    ab.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == "evaluate-policy":
        readygary = None
        if args.readygary_json:
            readygary = json.loads(Path(args.readygary_json).read_text(encoding="utf-8"))
        bundle = evaluate_policy(
            Path(args.twin_state),
            args.policy,
            Path(args.output),
            schema_dir=Path(args.schema_dir) if args.schema_dir else None,
            seed=args.seed,
            readygary=readygary,
        )
        print(json.dumps({"wrote": args.output, "policy": bundle["policy_name"]}, indent=2))
        return 0

    if args.cmd == "validate-decision":
        result = validate_decision(
            Path(args.path),
            schema_dir=Path(args.schema_dir) if args.schema_dir else None,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "benchmark":
        rows = run_benchmark(
            Path(args.twin_state),
            repetitions=args.repetitions,
            warmup=args.warmup,
            seed=args.seed,
            schema_dir=Path(args.schema_dir) if args.schema_dir else None,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                flat["runtime_s_list"] = json.dumps(row["runtime_s_list"])
                writer.writerow(flat)
        print(str(out))
        return 0

    if args.cmd == "ablation":
        rows = run_ablations(
            Path(args.twin_state),
            seed=args.seed,
            schema_dir=Path(args.schema_dir) if args.schema_dir else None,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(str(out))
        return 0

    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

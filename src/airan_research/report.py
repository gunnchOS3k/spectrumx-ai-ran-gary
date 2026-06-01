"""Export AI-RAN toy reports to results/e2e/."""
from __future__ import annotations

import json
from pathlib import Path


def write_reports(payload: dict, root: Path) -> tuple[Path, Path, Path]:
    e2e = root / "results" / "e2e"
    e2e.mkdir(parents=True, exist_ok=True)
    json_path = e2e / "airan_policy_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = e2e / "airan_policy_demo.md"
    lines = ["# AI-RAN Toy Policy Demo", "", "## Baseline"]
    for k, v in payload.get("baseline", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## AI policy"]
    for k, v in payload.get("ai_policy", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", f"_{payload.get('note', '')}_"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = e2e / "airan_policy_table.md"
    table.write_text(
        "| Metric | Baseline | AI policy |\n|--------|----------|----------|\n"
        + "\n".join(
            f"| {k} | {payload['baseline'].get(k)} | {payload['ai_policy'].get(k)} |"
            for k in payload.get("baseline", {})
        )
        + "\n",
        encoding="utf-8",
    )
    return md, json_path, table

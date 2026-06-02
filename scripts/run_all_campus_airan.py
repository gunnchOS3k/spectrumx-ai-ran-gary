#!/usr/bin/env python3
import json
from pathlib import Path
import yaml

cfg = Path(__file__).resolve().parents[1] / "configs" / "campus_ai_ran_profiles"
out = Path("results/campus_airan")
out.mkdir(parents=True, exist_ok=True)
for p in sorted(cfg.glob("*.yaml")):
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    sid = data["site_id"]
    metrics = {"fairness_stub": 0.7, "energy_stub": 0.6, "evidence_status": data.get("evidence_status", "smoke_test_only")}
    (out / f"{sid}_airan_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / f"{sid}_airan_policy_report.md").write_text(
        f"# AI-RAN campus profile — {sid}\n\nSmoke scenario only. Competition evaluate() path unchanged.\n", encoding="utf-8"
    )
print("Wrote campus AI-RAN results")

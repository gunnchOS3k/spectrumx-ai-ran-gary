#!/usr/bin/env python3
"""
Best-effort AODT / Aerial bootstrap helper. Stays honest when the machine cannot complete an install.

Environment (flags only, no secrets in this file):
  - ENABLE_AODT_BOOTSTRAP=1  — run extended checks (docker, optional pull)
  - ENABLE_DOCKER_PULLS=1    — if bootstrap enabled, attempt a small nvcr.io pull (may fail offline)

Always writes/updates data/aerial_omniverse/access_summary.json via the same logic as check_ngc_access.
Optionally writes bootstrap_report.json with machine-readable status (no secrets).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "aerial_omniverse"
REPORT_FILE = OUT_DIR / "bootstrap_report.json"


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stderr or p.stdout or "")[:2000]
    except Exception as e:
        return 99, f"{type(e).__name__}: {e}"


def _load_check_ngc():
    import importlib.util

    p = REPO_ROOT / "scripts" / "check_ngc_access.py"
    spec = importlib.util.spec_from_file_location("check_ngc_access", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ck = _load_check_ngc()
    access = ck.build_summary()
    ACCESS_FILE = ck.OUT_FILE
    ACCESS_FILE.write_text(json.dumps(access, indent=2), encoding="utf-8")

    bootstrap_on = os.environ.get("ENABLE_AODT_BOOTSTRAP", "").strip() == "1"
    pull_on = os.environ.get("ENABLE_DOCKER_PULLS", "").strip() == "1"

    report: dict = {
        "bootstrap_report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enable_aodt_bootstrap": bootstrap_on,
        "enable_docker_pulls": pull_on,
        "steps": [],
        "outcome": "access_summary_only",
    }

    if not bootstrap_on:
        report["steps"].append(
            {"step": "bootstrap", "status": "skipped", "reason": "ENABLE_AODT_BOOTSTRAP is not set to 1"}
        )
        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {ACCESS_FILE.relative_to(REPO_ROOT)} (bootstrap skipped)")
        print(f"Wrote {REPORT_FILE.relative_to(REPO_ROOT)}")
        return 0

    report["steps"].append({"step": "docker_version", "status": "running"})
    code, out = _run(["docker", "version"], timeout=60)
    report["steps"][-1] = {"step": "docker_version", "status": "ok" if code == 0 else "failed", "detail": out[:500]}

    if pull_on and code == 0:
        report["steps"].append({"step": "docker_pull_smoke", "status": "running"})
        # Small public-ish CUDA base — may still fail without auth/network; never embed credentials
        img = "nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04"
        c2, o2 = _run(["docker", "pull", img], timeout=600)
        report["steps"][-1] = {
            "step": "docker_pull_smoke",
            "status": "ok" if c2 == 0 else "failed",
            "image": img,
            "detail": o2[:800],
        }
    elif pull_on:
        report["steps"].append(
            {"step": "docker_pull_smoke", "status": "skipped", "reason": "docker_version failed"}
        )

    report["outcome"] = "bootstrap_partial"
    report["notes"] = (
        "Full AODT / Omniverse Digital Twin install is outside this script. "
        "Use NVIDIA documentation + NGC + local GPU. This report is diagnostic only."
    )
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {ACCESS_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify E2E required files exist for spectrumx-ai-ran-gary."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "docs/END_TO_END_READINESS.md",
    "docs/EXTERNAL_RESEARCHER_QUICKSTART.md",
    "quality/E2E_READINESS_CHECKLIST.md",
    "reproducibility/E2E_RUN_RECORD.md",
    "Makefile",
    "results/e2e/airan_policy_demo.md",
    "results/e2e/airan_policy_demo.json",
]

def main():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("FAIL missing:", *missing, sep="\n  ")
        return 1
    if "e2e:" not in (ROOT / "Makefile").read_text():
        print("FAIL: Makefile missing e2e target")
        return 1
    readme = (ROOT / "README.md").read_text(errors="ignore")
    if "End-to-End Research Artifact" not in readme:
        print("FAIL: README missing E2E section")
        return 1
    print("PASS all required E2E artifacts for spectrumx-ai-ran-gary")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "results/e2e/airan_policy_demo.md",
    "results/e2e/airan_policy_metrics.json",
    "results/e2e/airan_policy_table.md",
    "docs/video_walkthrough_script.md",
    "quality/REALITY_AUDIT.md",
    "paper/implementation.md",
    "docs/diagrams/code_path_main_demo.mmd",
]
def main():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]

    for d in ["context.mmd", "sequence_main_demo.mmd", "code_path_main_demo.mmd"]:
        if not (ROOT / "docs/diagrams" / d).exists():
            missing.append("docs/diagrams/" + d)
    if missing:
        print("FAIL", missing)
        return 1
    print("PASS e2e artifacts spectrumx-ai-ran-gary")
    return 0
if __name__ == "__main__":
    sys.exit(main())

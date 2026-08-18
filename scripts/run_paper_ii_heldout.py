#!/usr/bin/env python3
"""Held-out confirmatory split. Protocol must exist (committed first)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_paper_ii_digital import main as digital_main


def main() -> int:
    sys.argv = [sys.argv[0], "--held-out", "--domain-shift", "--ablations"]
    return digital_main()


if __name__ == "__main__":
    raise SystemExit(main())

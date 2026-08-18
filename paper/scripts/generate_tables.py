#!/usr/bin/env python3
"""Tabulate toy AI-RAN demo if present; otherwise RESULT_PENDING."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
DEST = OUT / "rq2_toy.tex"
candidates = [
    ROOT / "results" / "e2e" / "airan_toy.json",
    ROOT / "results" / "demo_airan_policy.json",
]


def find() -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    for p in (ROOT / "results").rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("mode") == "toy" and "baseline" in data:
            return p
    return None


def main() -> int:
    src = find()
    if src is None:
        DEST.write_text(
            "\\textbf{RESULT\\_PENDING.} Run \\texttt{python3 scripts/demo\\_airan\\_policy.py --toy} first.\\par\n",
            encoding="utf-8",
        )
        print("RESULT_PENDING", DEST)
        return 0
    data = json.loads(src.read_text(encoding="utf-8"))
    note = data.get("note", "toy")
    DEST.write_text(
        "\\begin{table}[h]\\centering\n"
        "\\caption{Toy policy snapshot (SYNTHETIC\\_SIM; not competition IQ).}\n"
        "\\begin{tabular}{ll}\\toprule field & value \\\\\\midrule\n"
        f"mode & {data.get('mode')} \\\\\nseed & {data.get('seed')} \\\\\n"
        f"note & {note} \\\\\\bottomrule\\end{{tabular}}\\end{{table}}\n",
        encoding="utf-8",
    )
    print("wrote", DEST, "from", src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

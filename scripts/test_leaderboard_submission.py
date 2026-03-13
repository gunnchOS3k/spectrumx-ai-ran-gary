"""
Local smoke test for the leaderboard_baseline_v1 submission.

Usage (from repo root):
    python scripts/test_leaderboard_submission.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np


def _find_candidate_npy_files() -> List[Path]:
    """
    Look for small synthetic/demo .npy files, falling back to an empty list.

    We deliberately avoid touching real competition dataset paths and only
    look under typical synthetic/demo locations.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates: List[Path] = []

    search_dirs = [
        repo_root / "data" / "synth_gary_twin",
        repo_root / "data" / "synthetic" / "gary_micro_twin",
    ]

    for d in search_dirs:
        if d.is_dir():
            for path in d.glob("*.npy"):
                candidates.append(path)

    return candidates


def _create_temp_sample() -> Path:
    """
    Create a minimal synthetic IQ sample in the repo root and return its path.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tmp_path = repo_root / "tmp_leaderboard_test_sample.npy"

    sample_rate = 1_000_000
    duration = 0.01  # 10 ms, smaller than full 1 second to keep it lightweight
    n_samples = int(sample_rate * duration)

    rng = np.random.default_rng(12345)
    noise_i = rng.normal(0.0, 0.1, size=n_samples)
    noise_q = rng.normal(0.0, 0.1, size=n_samples)

    iq = noise_i + 1j * noise_q
    np.save(tmp_path, iq.astype(np.complex64))

    return tmp_path


def main() -> int:
    # Import evaluate from the submission package
    try:
        from submissions.leaderboard_baseline_v1.main import evaluate
    except Exception as e:  # pragma: no cover - import failure path
        print("FAIL: could not import submissions.leaderboard_baseline_v1.main.evaluate")
        print(f"Error: {e}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    npy_files = _find_candidate_npy_files()

    if not npy_files:
        tmp = _create_temp_sample()
        npy_files = [tmp]
        print(f"No synthetic .npy files found; created temp sample: {tmp.relative_to(repo_root)}")

    passed = True

    for path in npy_files[:3]:  # limit to a few files
        rel = path.relative_to(repo_root)
        try:
            result = evaluate(str(path))
        except Exception as e:
            print(f"FAIL: evaluate() raised on {rel}: {e}")
            passed = False
            continue

        if not isinstance(result, int):
            print(f"FAIL: evaluate({rel}) returned non-int type: {type(result)}")
            passed = False
        elif result not in (0, 1):
            print(f"FAIL: evaluate({rel}) returned invalid value: {result} (expected 0 or 1)")
            passed = False
        else:
            print(f"PASS: evaluate({rel}) -> {result}")

    if passed:
        print("OVERALL RESULT: PASS")
        return 0
    else:
        print("OVERALL RESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())


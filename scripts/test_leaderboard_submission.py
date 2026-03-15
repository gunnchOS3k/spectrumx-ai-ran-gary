"""
Local smoke test for the leaderboard_baseline_v1 submission.

Usage (from repo root):
    python scripts/test_leaderboard_submission.py
"""

from __future__ import annotations

import sys
import tempfile
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


def _create_1s_complex64_sample(out_dir: Path) -> Path:
    """Create a 1-second synthetic IQ sample, complex64 shape (N,)."""
    path = out_dir / "tmp_leaderboard_1s_complex64.npy"
    n_samples = 1_000_000  # 1 s at 1 MHz
    rng = np.random.default_rng(42)
    iq = rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)
    np.save(path, iq.astype(np.complex64))
    return path


def _create_float_n2_sample(out_dir: Path) -> Path:
    """Create a synthetic IQ sample as (N, 2) float I/Q."""
    path = out_dir / "tmp_leaderboard_float_n2.npy"
    n_samples = 1_000_000
    rng = np.random.default_rng(99)
    iq_float = np.stack([rng.normal(0, 0.1, n_samples), rng.normal(0, 0.1, n_samples)], axis=1)
    np.save(path, iq_float.astype(np.float32))
    return path


def main() -> int:
    # Import evaluate from the submission package
    try:
        from submissions.leaderboard_baseline_v1.main import evaluate
    except Exception as e:  # pragma: no cover - import failure path
        print("FAIL: could not import submissions.leaderboard_baseline_v1.main.evaluate")
        print(f"Error: {e}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    # Always create and test 1-second complex64 and (N,2) float samples (use temp dir)
    tmp_dir = Path(tempfile.mkdtemp(prefix="leaderboard_test_"))
    p_1s = _create_1s_complex64_sample(tmp_dir)
    created_paths.append(p_1s)
    p_n2 = _create_float_n2_sample(tmp_dir)
    created_paths.append(p_n2)
    npy_files = [p_1s, p_n2] + _find_candidate_npy_files()[:1]

    passed = True

    for path in npy_files[:5]:  # our 2 required + up to 1 from disk
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = Path(path.name)
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

    # Remove temp files and temp dir
    for p in [p_1s, p_n2]:
        if p.exists():
            p.unlink()
    if tmp_dir.exists():
        tmp_dir.rmdir()
    if passed:
        print("OVERALL RESULT: PASS")
        return 0
    else:
        print("OVERALL RESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())


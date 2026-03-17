"""
Local validation for the feature-based leaderboard submission.

Generates a few synthetic IQ samples and runs:
    - submissions.leaderboard_baseline_v1.main.evaluate(filename)
Also prints feature vectors and raw linear scores (when model artifact exists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from submissions.leaderboard_baseline_v1.main import (
    DEFAULT_SAMPLE_RATE,
    _extract_features,
    _load_iq_auto,
    evaluate,
)


def _make_noise_sample(path: Path) -> None:
    n = int(DEFAULT_SAMPLE_RATE)
    rng = np.random.default_rng(0)
    iq = (
        rng.normal(0.0, 0.1, n)
        + 1j * rng.normal(0.0, 0.1, n)
    ).astype(np.complex64)
    np.save(path, iq)


def _make_signal_sample(path: Path) -> None:
    n = int(DEFAULT_SAMPLE_RATE)
    rng = np.random.default_rng(1)
    t = np.arange(n) / DEFAULT_SAMPLE_RATE
    noise = (
        rng.normal(0.0, 0.1, n)
        + 1j * rng.normal(0.0, 0.1, n)
    )
    tone = 0.5 * np.exp(1j * 2 * np.pi * 100_000 * t)
    iq = (noise + tone).astype(np.complex64)
    np.save(path, iq)


def _make_float_n2_sample(path: Path) -> None:
    n = int(DEFAULT_SAMPLE_RATE)
    rng = np.random.default_rng(2)
    iq_float = np.stack(
        [rng.normal(0.0, 0.1, n), rng.normal(0.0, 0.1, n)],
        axis=1,
    ).astype(np.float32)
    np.save(path, iq_float)


def _load_artifact(artifact_path: Path):
    if not artifact_path.is_file():
        return None
    data = np.load(artifact_path, allow_pickle=True)
    weights = data["weights"].astype(np.float64)
    bias = float(data["bias"])
    mean = data["mean"].astype(np.float64)
    std = data["std"].astype(np.float64)
    threshold = float(data["threshold"])
    return weights, bias, mean, std, threshold


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    noise_path = repo_root / "val_noise_1s_complex64.npy"
    signal_path = repo_root / "val_signal_1s_complex64.npy"
    float_path = repo_root / "val_iq_float_n2.npy"

    _make_noise_sample(noise_path)
    _make_signal_sample(signal_path)
    _make_float_n2_sample(float_path)

    artifact_path = repo_root / "submissions" / "leaderboard_baseline_v1" / "feature_detector_v1.npz"
    artifact = _load_artifact(artifact_path)
    if artifact is None:
        print(f"Warning: model artifact not found at {artifact_path}, will only show features and evaluate() outputs.")
    else:
        weights, bias, mean, std, threshold = artifact

    for p in [noise_path, signal_path, float_path]:
        iq = _load_iq_auto(p)
        feats = _extract_features(iq, sample_rate=DEFAULT_SAMPLE_RATE)
        feat_names = sorted(feats.keys())
        x = np.asarray([feats[n] for n in feat_names], dtype=np.float64)

        print(f"\nFile: {p.name}")
        print("First 8 features:")
        for name in feat_names[:8]:
            print(f"  {name}: {feats[name]:.6f}")

        y = evaluate(str(p))
        print(f"evaluate() -> {y} (type={type(y)})")

        if artifact is not None and x.size == weights.size:
            xn = (x - mean) / std
            score = float(np.dot(weights, xn) + bias)
            print(f"Raw linear score: {score:.6f}, threshold: {threshold:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


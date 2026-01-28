"""
Run Phase 1 baselines (Energy, Spectral Flatness, PSD+LogReg).

Usage (from repo root):
  python scripts/run_phase1_baselines.py --smoke
  python scripts/run_phase1_baselines.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure repo root is on sys.path for "src.*" imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge_ran_gary.config import BaselineConfig, SpectrumXDatasetConfig
from src.edge_ran_gary.data_pipeline.spectrumx_loader import SpectrumXDataset
from src.edge_ran_gary.detection.baselines import (
    EnergyDetector,
    PSDLogRegDetector,
    SpectralFlatnessDetector,
)
from src.edge_ran_gary.sim.evaluation import Evaluator


def _make_synthetic(samples: int, n: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(42)
    X = []
    y = []
    for i in range(samples):
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        label = 0 if i % 2 == 0 else 1
        X.append(iq)
        y.append(label)
    return X, np.array(y, dtype=int)


def _load_dataset(cfg: SpectrumXDatasetConfig, download: bool) -> tuple[list[np.ndarray], np.ndarray]:
    ds = SpectrumXDataset(cfg)
    if download:
        ds.download(overwrite=False)
    if not ds.labeled_dir.exists():
        raise RuntimeError(
            "Labeled dataset not found. Set SDS_SECRET_TOKEN in .env and run with --download."
        )
    X, y, _ = ds.load_labeled()
    return X, y


def _evaluate_baseline(name: str, y_true: np.ndarray, preds: list[int], probs: list[float]):
    evaluator = Evaluator()
    results = evaluator.evaluate(y_true=y_true, y_pred=preds, y_proba=probs)
    print(f"\n{name}")
    for key, value in results.items():
        if value is None:
            print(f"  {key:10s}: None")
        else:
            print(f"  {key:10s}: {value:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline detectors.")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic data only.")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--sample-rate", type=float, default=BaselineConfig.sample_rate)
    parser.add_argument("--energy-threshold", type=float, default=BaselineConfig.energy_threshold)
    parser.add_argument(
        "--flatness-threshold",
        type=float,
        default=BaselineConfig.spectral_flatness_threshold,
    )
    args = parser.parse_args()

    if args.smoke:
        X, y = _make_synthetic(samples=40, n=1_000_000)
    else:
        X, y = _load_dataset(SpectrumXDatasetConfig(), download=args.download)

    if args.limit is not None:
        X = X[: args.limit]
        y = y[: args.limit]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )

    # Energy detector
    energy = EnergyDetector(threshold=args.energy_threshold)
    energy_preds = []
    energy_probs = []
    for x in X_val:
        pred, conf, _ = energy.predict(x)
        energy_preds.append(pred)
        energy_probs.append(conf)
    _evaluate_baseline("Energy Detector", y_val, energy_preds, energy_probs)

    # Spectral flatness detector
    flat = SpectralFlatnessDetector(
        threshold=args.flatness_threshold, sample_rate=args.sample_rate
    )
    flat_preds = []
    flat_probs = []
    for x in X_val:
        pred, conf, _ = flat.predict(x)
        flat_preds.append(pred)
        flat_probs.append(conf)
    _evaluate_baseline("Spectral Flatness Detector", y_val, flat_preds, flat_probs)

    # PSD + Logistic Regression
    psd = PSDLogRegDetector()
    psd.fit(X_train, y_train, sample_rate=args.sample_rate)
    psd_preds = []
    psd_probs = []
    for x in X_val:
        pred, prob = psd.predict(x, sample_rate=args.sample_rate)
        psd_preds.append(pred)
        psd_probs.append(prob)
    _evaluate_baseline("PSD + Logistic Regression", y_val, psd_preds, psd_probs)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

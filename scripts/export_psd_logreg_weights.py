"""
Train PSD+LogReg on labeled data and export weights for submission.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.edge_ran_gary.detection.baselines import PSDLogRegDetector


def load_iq(path: Path, max_samples: int | None = None) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if np.iscomplexobj(data):
        iq = data[:max_samples] if max_samples else data
        if iq.dtype == np.complex64:
            return np.array(iq, copy=False)
        return iq.astype(np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        i = data[:, 0][:max_samples] if max_samples else data[:, 0]
        q = data[:, 1][:max_samples] if max_samples else data[:, 1]
        iq = i.astype(np.float32) + 1j * q.astype(np.float32)
        return iq.astype(np.complex64)
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        if max_samples:
            data = data[: max_samples * 2]
        i = data[0::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        iq = (i + 1j * q).astype(np.complex64)
        return iq[:max_samples] if max_samples else iq
    raise ValueError(f"Unsupported IQ format: shape={data.shape}, dtype={data.dtype}")


def load_labeled(dataset_root: Path, max_samples: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    files_root = dataset_root / "files"
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()] if files_root.exists() else []
    if not user_dirs:
        raise FileNotFoundError("No user directory found under competition_dataset/files")

    user_dir = user_dirs[0]
    labeled_dir = user_dir / "VLA_brutal"
    training_dir = user_dir / "trainingData"
    gt_path = labeled_dir / "groundtruth.csv"
    if not labeled_dir.exists() or not gt_path.exists():
        raise FileNotFoundError("Expected VLA_brutal/groundtruth.csv not found.")
    if not training_dir.exists():
        raise FileNotFoundError("Expected trainingData folder not found.")

    gt = pd.read_csv(gt_path)
    X, y = [], []
    missing = []
    for _, row in gt.iterrows():
        filename = row["filename"]
        # Prefer trainingData, fallback to VLA_brutal if missing
        file_path = training_dir / filename
        if not file_path.exists():
            file_path = labeled_dir / filename
        if not file_path.exists():
            missing.append(filename)
            continue
        X.append(load_iq(file_path, max_samples=max_samples))
        y.append(int(row["label"]))

    if missing:
        print(f"Warning: {len(missing)} labeled files missing; training on available files only.")

    return X, np.array(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PSD+LogReg weights for submission.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
        help="Path to competition_dataset",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate for PSD computation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/psd_logreg_weights.json"),
        help="Path to save weights JSON",
    )
    parser.add_argument(
        "--max-iq-samples",
        type=int,
        default=1_000_000,
        help="Max IQ samples per file for feature extraction (0 = all)",
    )
    args = parser.parse_args()

    max_samples = None if args.max_iq_samples == 0 else args.max_iq_samples
    X_iq, y = load_labeled(args.dataset_root, max_samples=max_samples)
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            "Need at least two label classes to train Logistic Regression. "
            f"Found labels: {unique.tolist()}"
        )

    detector = PSDLogRegDetector()
    detector.fit(X_iq, y, sample_rate=args.sample_rate)

    weights = detector.model.coef_.reshape(-1).tolist()
    bias = float(detector.model.intercept_[0])

    payload = {
        "sample_rate": float(args.sample_rate),
        "weights": weights,
        "bias": bias,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved weights to: {args.output}")
    print(payload)


if __name__ == "__main__":
    main()

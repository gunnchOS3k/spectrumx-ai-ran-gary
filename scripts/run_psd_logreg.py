"""
Train and evaluate PSD + Logistic Regression baseline on labeled data.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.edge_ran_gary.detection.baselines import PSDLogRegDetector


def load_iq(path: Path) -> np.ndarray:
    data = np.load(path)
    if np.iscomplexobj(data):
        return data.astype(np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        i = data[0::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)
    raise ValueError(f"Unsupported IQ format: shape={data.shape}, dtype={data.dtype}")


def load_labeled(dataset_root: Path) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    files_root = dataset_root / "files"
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()] if files_root.exists() else []
    if not user_dirs:
        raise FileNotFoundError("No user directory found under competition_dataset/files")

    user_dir = user_dirs[0]
    labeled_dir = user_dir / "VLA_brutal"
    gt_path = labeled_dir / "groundtruth.csv"
    if not labeled_dir.exists() or not gt_path.exists():
        raise FileNotFoundError("Expected VLA_brutal/groundtruth.csv not found.")

    gt = pd.read_csv(gt_path)
    X, y, filenames = [], [], []
    for _, row in gt.iterrows():
        file_path = labeled_dir / row["filename"]
        X.append(load_iq(file_path))
        y.append(int(row["label"]))
        filenames.append(row["filename"])

    return X, np.array(y), filenames


def main() -> None:
    parser = argparse.ArgumentParser(description="PSD + Logistic Regression baseline.")
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
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/psd_logreg_metrics.json"),
        help="Path to save metrics JSON",
    )
    args = parser.parse_args()

    X_iq, y, filenames = load_labeled(args.dataset_root)
    X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
        X_iq,
        y,
        filenames,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    detector = PSDLogRegDetector()
    detector.fit(X_train, y_train, sample_rate=args.sample_rate)

    probs = []
    preds = []
    for iq in X_test:
        pred, prob = detector.predict(iq, sample_rate=args.sample_rate)
        preds.append(pred)
        probs.append(prob)

    metrics = {
        "dataset_root": str(args.dataset_root),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "sample_rate": float(args.sample_rate),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    except ValueError:
        metrics["roc_auc"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))

    print(f"Saved metrics to: {args.output}")
    print(metrics)


if __name__ == "__main__":
    main()

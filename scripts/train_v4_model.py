"""
Train PSD+LogReg with accuracy-tuned threshold and pseudo-labeling.

Steps:
1. Train on labeled data (VLA_brutal/groundtruth.csv)
2. Sweep probability threshold to maximize accuracy (not F1)
3. Pseudo-label high-confidence unlabeled samples (trainingData)
4. Retrain on labeled + pseudo-labeled data
5. Re-sweep threshold
6. Export final weights, bias, and optimal threshold
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        return (i.astype(np.float32) + 1j * q.astype(np.float32)).astype(np.complex64)
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        if max_samples:
            data = data[: max_samples * 2]
        i = data[0::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        iq = (i + 1j * q).astype(np.complex64)
        return iq[:max_samples] if max_samples else iq
    raise ValueError(f"Unsupported IQ format: shape={data.shape}, dtype={data.dtype}")


def extract_features(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    _, psd = signal.welch(
        iq, fs=sample_rate, nperseg=1024, return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    if psd_mag.size == 0:
        return np.zeros(10, dtype=float)

    psd_pos = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    arith_mean = float(np.mean(psd_mag))
    flatness = geo_mean / arith_mean if arith_mean > 0 else 0.0

    mag = np.abs(iq)
    power = mag ** 2

    return np.array([
        np.mean(psd_mag),
        np.std(psd_mag),
        np.max(psd_mag),
        np.min(psd_mag),
        np.percentile(psd_mag, 25),
        np.percentile(psd_mag, 75),
        flatness,
        float(np.mean(power)),
        float(np.std(mag)),
        float(np.max(psd_mag) / (arith_mean + 1e-30)),
    ], dtype=float)


def load_labeled(dataset_root: Path, max_samples: int | None) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    files_root = dataset_root / "files"
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()] if files_root.exists() else []
    user_dir = user_dirs[0]
    labeled_dir = user_dir / "VLA_brutal"
    training_dir = user_dir / "trainingData"
    gt_path = labeled_dir / "groundtruth.csv"

    gt = pd.read_csv(gt_path)
    X, y, fnames = [], [], []
    for _, row in gt.iterrows():
        filename = row["filename"]
        file_path = training_dir / filename
        if not file_path.exists():
            file_path = labeled_dir / filename
        if not file_path.exists():
            continue
        X.append(load_iq(file_path, max_samples=max_samples))
        y.append(int(row["label"]))
        fnames.append(filename)
    return X, np.array(y), fnames


def load_unlabeled(dataset_root: Path, max_samples: int | None, exclude: set[str]) -> tuple[list[np.ndarray], list[Path]]:
    files_root = dataset_root / "files"
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()] if files_root.exists() else []
    user_dir = user_dirs[0]
    training_dir = user_dir / "trainingData"

    X, paths = [], []
    if training_dir.exists():
        for npy in sorted(training_dir.glob("*.npy")):
            if npy.name in exclude:
                continue
            X.append(load_iq(npy, max_samples=max_samples))
            paths.append(npy)
    return X, paths


def sweep_threshold_for_accuracy(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    best_acc = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.01, 0.99, 200):
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v4 model with pseudo-labeling.")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
    )
    parser.add_argument("--sample-rate", type=float, default=1e6)
    parser.add_argument("--max-iq-samples", type=int, default=1_000_000)
    parser.add_argument("--pseudo-confidence", type=float, default=0.85,
                        help="Min confidence for pseudo-label inclusion")
    parser.add_argument("--output", type=Path, default=Path("results/v4_model.json"))
    args = parser.parse_args()

    max_samples = None if args.max_iq_samples == 0 else args.max_iq_samples

    # --- Step 1: Load labeled data ---
    print("Loading labeled data...")
    X_iq_labeled, y_labeled, fnames_labeled = load_labeled(args.dataset_root, max_samples)
    print(f"Labeled: {len(X_iq_labeled)} files | Classes: {dict(zip(*np.unique(y_labeled, return_counts=True)))}")

    print("Extracting features from labeled data...")
    X_labeled = np.array([extract_features(iq, args.sample_rate) for iq in X_iq_labeled])

    # --- Step 2: Train initial model ---
    print("Training initial LogReg...")
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    model.fit(X_labeled, y_labeled)

    probs_labeled = model.predict_proba(X_labeled)[:, 1]
    thr_init, acc_init = sweep_threshold_for_accuracy(probs_labeled, y_labeled)
    print(f"Initial train accuracy: {acc_init:.3f} at threshold {thr_init:.4f}")

    # Cross-val accuracy estimate
    cv_probs = cross_val_predict(
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        X_labeled, y_labeled, cv=5, method="predict_proba",
    )[:, 1]
    thr_cv, acc_cv = sweep_threshold_for_accuracy(cv_probs, y_labeled)
    print(f"5-fold CV accuracy: {acc_cv:.3f} at threshold {thr_cv:.4f}")

    # --- Step 3: Pseudo-label unlabeled data ---
    print("Loading unlabeled data...")
    exclude_set = set(fnames_labeled)
    X_iq_unlabeled, paths_unlabeled = load_unlabeled(args.dataset_root, max_samples, exclude_set)
    print(f"Unlabeled: {len(X_iq_unlabeled)} files")

    if X_iq_unlabeled:
        print("Extracting features from unlabeled data...")
        X_unlabeled = np.array([extract_features(iq, args.sample_rate) for iq in X_iq_unlabeled])

        probs_unlabeled = model.predict_proba(X_unlabeled)[:, 1]
        high_conf_pos = probs_unlabeled >= args.pseudo_confidence
        high_conf_neg = probs_unlabeled <= (1.0 - args.pseudo_confidence)
        high_conf = high_conf_pos | high_conf_neg

        pseudo_X = X_unlabeled[high_conf]
        pseudo_y = (probs_unlabeled[high_conf] >= 0.5).astype(int)
        n_pseudo_pos = int(pseudo_y.sum())
        n_pseudo_neg = int(len(pseudo_y) - n_pseudo_pos)
        print(f"Pseudo-labeled: {len(pseudo_y)} (pos={n_pseudo_pos}, neg={n_pseudo_neg}) "
              f"out of {len(X_iq_unlabeled)} unlabeled")

        # --- Step 4: Retrain with augmented data ---
        if len(pseudo_y) > 0:
            X_aug = np.vstack([X_labeled, pseudo_X])
            y_aug = np.concatenate([y_labeled, pseudo_y])
            print(f"Augmented training set: {len(y_aug)} total")

            model_final = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            model_final.fit(X_aug, y_aug)

            probs_labeled_final = model_final.predict_proba(X_labeled)[:, 1]
            thr_final, acc_final = sweep_threshold_for_accuracy(probs_labeled_final, y_labeled)
            print(f"Final model accuracy on labeled: {acc_final:.3f} at threshold {thr_final:.4f}")
        else:
            print("No high-confidence pseudo-labels; using initial model.")
            model_final = model
            thr_final = thr_cv
            acc_final = acc_cv
    else:
        model_final = model
        thr_final = thr_cv
        acc_final = acc_cv

    # --- Step 5: Export ---
    weights = model_final.coef_.reshape(-1).tolist()
    bias = float(model_final.intercept_[0])

    payload = {
        "sample_rate": float(args.sample_rate),
        "num_features": len(weights),
        "weights": weights,
        "bias": bias,
        "optimal_threshold": float(thr_final),
        "labeled_accuracy": float(acc_final),
        "cv_accuracy": float(acc_cv),
        "cv_threshold": float(thr_cv),
        "pseudo_labeled_count": int(len(pseudo_y)) if X_iq_unlabeled else 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved v4 model to: {args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

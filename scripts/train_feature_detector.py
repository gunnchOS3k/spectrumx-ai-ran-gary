"""
Train a lightweight feature-based detector for 1-second IQ windows.

This script is for LOCAL/OFFLINE use only. It is NOT required by the
leaderboard environment. The resulting model parameters can be exported
to a tiny .npz artifact and consumed by the submission wrapper.

High-level steps:
    1. Load labeled IQ samples from a directory + CSV (competition subset).
    2. Extract handcrafted features per sample (same as submission wrapper).
    3. Train and compare:
        - Logistic Regression
        - Linear SVM
        - (Optional) Decision Tree
    4. Report metrics and choose the best model.
    5. Export feature normalization + linear weights + bias + threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from submissions.leaderboard_baseline_v1.main import _extract_features, _load_iq_auto, DEFAULT_SAMPLE_RATE


@dataclass
class ModelExport:
    name: str
    weights: np.ndarray
    bias: float
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    threshold: float


def _load_dataset(iq_dir: Path, metadata_csv: Path, file_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    paths: List[Path] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        fname = str(row[file_col])
        label = int(row[label_col])
        p = iq_dir / fname
        if not p.is_file():
            continue
        paths.append(p)
        labels.append(label)

    X_features: List[np.ndarray] = []
    y = np.asarray(labels, dtype=int)
    for p in paths:
        iq = _load_iq_auto(p)
        feats_dict: Dict[str, float] = _extract_features(iq, sample_rate=DEFAULT_SAMPLE_RATE)
        if not X_features:
            feature_names = sorted(feats_dict.keys())
        feats_vec = np.asarray([feats_dict[n] for n in feature_names], dtype=np.float64)
        X_features.append(feats_vec)

    X = np.vstack(X_features)
    return X, y


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    return Xn, mean, std


def _train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[ModelExport, Dict[str, Dict[str, float]]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "linear_svm": LinearSVC(),
    }

    metrics_per_model: Dict[str, Dict[str, float]] = {}
    best_export: ModelExport | None = None
    best_f1 = -1.0

    for name, clf in models.items():
        y_true_all: List[int] = []
        scores_all: List[float] = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            X_tr_n, mean, std = _standardize(X_tr)
            X_val_n = (X_val - mean) / std

            clf.fit(X_tr_n, y_tr)

            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_val_n)
            elif hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(X_val_n)[:, 1]
            else:
                scores = clf.predict(X_val_n)

            y_true_all.extend(y_val.tolist())
            scores_all.extend(scores.tolist())

        y_true = np.asarray(y_true_all, dtype=int)
        scores = np.asarray(scores_all, dtype=np.float64)

        # Choose threshold to increase TP while keeping FP low.
        thresholds = np.linspace(np.min(scores), np.max(scores), num=100)
        best_local_f1 = -1.0
        best_local_thr = 0.0
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            # Prefer configurations with low FP; skip if FP is too large fraction.
            if fp > 0.2 * (tn + fp + 1e-9):
                continue
            f1 = metrics.f1_score(y_true, y_pred)
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_thr = thr

        y_pred_best = (scores >= best_local_thr).astype(int)
        acc = metrics.accuracy_score(y_true, y_pred_best)
        prec = metrics.precision_score(y_true, y_pred_best, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred_best, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred_best)
        cm = metrics.confusion_matrix(y_true, y_pred_best, labels=[0, 1])

        metrics_per_model[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": float(best_local_thr),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        if f1 > best_f1:
            best_f1 = f1
            # Retrain on full data with standardization
            Xn, mean_full, std_full = _standardize(X)
            clf.fit(Xn, y)

            if hasattr(clf, "coef_"):
                weights = clf.coef_.ravel().astype(np.float64)
                bias = float(clf.intercept_.ravel()[0])
            else:
                # Fallback: approximate by using a shallow copy of thresholds.
                weights = np.ones(X.shape[1], dtype=np.float64)
                bias = float(-best_local_thr)

            feature_names = [f"f{i}" for i in range(X.shape[1])]

            best_export = ModelExport(
                name=name,
                weights=weights,
                bias=bias,
                feature_names=feature_names,
                mean=mean_full.astype(np.float64),
                std=std_full.astype(np.float64),
                threshold=float(best_local_thr),
            )

    assert best_export is not None
    return best_export, metrics_per_model


def _save_export(export: ModelExport, out_path: Path) -> None:
    np.savez(
        out_path,
        name=export.name,
        weights=export.weights,
        bias=export.bias,
        feature_names=np.asarray(export.feature_names),
        mean=export.mean,
        std=export.std,
        threshold=export.threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train feature-based baseline detector.")
    parser.add_argument("--iq-dir", type=str, required=True, help="Directory with .npy IQ files.")
    parser.add_argument("--metadata", type=str, required=True, help="CSV with file and label columns.")
    parser.add_argument("--file-col", type=str, default="file", help="Column name for file names.")
    parser.add_argument("--label-col", type=str, default="label", help="Column name for labels (0/1).")
    parser.add_argument(
        "--out",
        type=str,
        default="submissions/leaderboard_baseline_v1/feature_detector_v1.npz",
        help="Output .npz path for exported model.",
    )
    args = parser.parse_args()

    iq_dir = Path(args.iq_dir)
    metadata_csv = Path(args.metadata)
    out_path = Path(args.out)

    X, y = _load_dataset(iq_dir, metadata_csv, file_col=args.file_col, label_col=args.label_col)

    export, metrics_per_model = _train_and_eval(X, y)

    print("Model comparison:")
    for name, m in metrics_per_model.items():
        print(
            f"{name}: "
            f"acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, "
            f"rec={m['recall']:.3f}, f1={m['f1']:.3f}, thr={m['threshold']:.3f}, "
            f"tn={m['tn']}, fp={m['fp']}, fn={m['fn']}, tp={m['tp']}"
        )

    print(f"\nBest model: {export.name} with F1={metrics_per_model[export.name]['f1']:.3f}")
    print(f"Exporting to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_export(export, out_path)


if __name__ == "__main__":
    main()


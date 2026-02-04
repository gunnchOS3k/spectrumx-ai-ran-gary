"""
Sweep thresholds for baseline detectors using labeled data
or estimate thresholds from unlabeled data percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class FileRecord:
    path: Path
    label: int | None


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


def spectral_flatness(iq: np.ndarray, sample_rate: float, nperseg: int = 1024) -> float:
    _, psd = signal.welch(
        iq,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density",
    )
    psd_mag = np.abs(psd)
    psd_mag = psd_mag[psd_mag > 0]
    if psd_mag.size == 0:
        return 0.0
    geometric_mean = np.exp(np.mean(np.log(psd_mag)))
    arithmetic_mean = np.mean(psd_mag)
    if arithmetic_mean == 0:
        return 0.0
    return float(geometric_mean / arithmetic_mean)


def mean_power(iq: np.ndarray) -> float:
    return float(np.mean(np.abs(iq) ** 2))


def collect_files(dataset_root: Path) -> tuple[list[FileRecord], list[FileRecord]]:
    files_root = dataset_root / "files"
    if not files_root.exists():
        return [], []

    # Dataset layout (user-confirmed):
    # - labeled data in VLA_brutal/ with groundtruth.csv
    # - unlabeled data in trainingData/
    labeled_folder_variants = ["VLA_brutal"]
    unlabeled_folder_variants = ["trainingData"]

    labeled_records: list[FileRecord] = []
    unlabeled_records: list[FileRecord] = []
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()]
    for user_dir in user_dirs:
        # labeled records come only from VLA_brutal with groundtruth.csv
        label_map: dict[str, int] = {}
        for fname in labeled_folder_variants:
            ldir = user_dir / fname
            gt_path = ldir / "groundtruth.csv"
            if ldir.exists() and gt_path.exists():
                df = pd.read_csv(gt_path)
                filename_col = next(
                    (c for c in df.columns if c.lower() in {"filename", "file", "fname", "path"}),
                    None,
                )
                label_col = next(
                    (c for c in df.columns if c.lower() in {"label", "y", "target"}),
                    None,
                )
                if filename_col and label_col:
                    label_map = {
                        Path(name).name: int(label) for name, label in zip(df[filename_col], df[label_col])
                    }
                for npy in sorted(ldir.glob("*.npy")):
                    label = label_map.get(npy.name)
                    if label is not None:
                        labeled_records.append(FileRecord(npy, label))

        # unlabeled records come only from trainingData/
        for fname in unlabeled_folder_variants:
            udir = user_dir / fname
            if udir.exists():
                for npy in sorted(udir.glob("*.npy")):
                    unlabeled_records.append(FileRecord(npy, None))

    return labeled_records, unlabeled_records


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, higher_is_positive: bool) -> tuple[float, float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 50)
    best_f1 = -1.0
    best_threshold = thresholds[0]
    for t in thresholds:
        if higher_is_positive:
            preds = (scores > t).astype(int)
        else:
            preds = (scores < t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    try:
        roc = roc_auc_score(labels, scores if higher_is_positive else -scores)
    except ValueError:
        roc = float("nan")

    return best_threshold, best_f1, roc


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep for baseline detectors.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
        help="Path to competition_dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["labeled", "unlabeled"],
        default="labeled",
        help="Use labels for sweep, or percentile from unlabeled data",
    )
    parser.add_argument(
        "--detector",
        choices=["energy", "flatness", "both"],
        default="both",
        help="Which detector to evaluate",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate for PSD computation",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile for unlabeled threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_thresholds.json"),
        help="Path to save threshold summary JSON",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to score (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when sampling files",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/baseline_metrics.json"),
        help="Path to save metrics summary JSON",
    )
    args = parser.parse_args()

    labeled_records, unlabeled_records = collect_files(args.dataset_root)
    if not labeled_records and not unlabeled_records:
        raise FileNotFoundError("No files found under competition_dataset/files")

    labeled = [r for r in labeled_records if r.label is not None]
    if args.mode == "labeled" and not labeled:
        raise ValueError("No labeled files with groundtruth.csv were found.")

    print(f"Labeled files: {len(labeled_records)} | Unlabeled files: {len(unlabeled_records)}")
    if labeled:
        label_values = np.array([r.label for r in labeled], dtype=int)
        unique, counts = np.unique(label_values, return_counts=True)
        label_dist = {str(k): int(v) for k, v in zip(unique, counts)}
        print("Label distribution:", label_dist)

    def compute_scores(metric_fn) -> tuple[np.ndarray, np.ndarray]:
        scores = []
        labels = []
        source = labeled if args.mode == "labeled" else unlabeled_records
        if args.max_files and len(source) > args.max_files:
            rng = np.random.default_rng(args.seed)
            source = list(rng.choice(source, size=args.max_files, replace=False))
        for record in source:
            iq = load_iq(record.path)
            scores.append(metric_fn(iq))
            if record.label is not None:
                labels.append(record.label)
        return np.array(scores, dtype=float), np.array(labels, dtype=int)

    summary: dict[str, dict[str, float | str | None]] = {
        "dataset_root": str(args.dataset_root),
        "mode": args.mode,
        "detector": args.detector,
    }

    def save_summary() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            try:
                existing = json.loads(args.output.read_text())
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}

        merged = dict(existing)
        for key, value in summary.items():
            merged[key] = value
        args.output.write_text(json.dumps(merged, indent=2))

    scores_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if args.detector in {"energy", "both"}:
        scores, labels = compute_scores(mean_power)
        scores_cache["energy"] = (scores, labels)
        if args.mode == "labeled":
            thr, best_f1, roc = sweep_thresholds(scores, labels, higher_is_positive=True)
            print(f"Energy threshold: {thr:.6f} | F1: {best_f1:.3f} | ROC-AUC: {roc:.3f}")
            summary["energy"] = {
                "threshold": float(thr),
                "best_f1": float(best_f1),
                "roc_auc": None if math.isnan(roc) else float(roc),
                "why_best": "Maximizes F1 score over a grid of thresholds on labeled data.",
            }
            save_summary()
        else:
            thr = float(np.percentile(scores, args.percentile))
            print(f"Energy threshold (p{args.percentile}): {thr:.6f}")
            summary["energy"] = {
                "threshold": float(thr),
                "percentile": float(args.percentile),
                "why_best": "Percentile of unlabeled metric distribution (assumes high energy indicates occupancy).",
            }
            save_summary()

    if args.detector in {"flatness", "both"}:
        scores, labels = compute_scores(lambda iq: spectral_flatness(iq, args.sample_rate))
        scores_cache["flatness"] = (scores, labels)
        if args.mode == "labeled":
            thr, best_f1, roc = sweep_thresholds(scores, labels, higher_is_positive=False)
            print(f"Flatness threshold: {thr:.6f} | F1: {best_f1:.3f} | ROC-AUC: {roc:.3f}")
            summary["flatness"] = {
                "threshold": float(thr),
                "best_f1": float(best_f1),
                "roc_auc": None if math.isnan(roc) else float(roc),
                "why_best": "Maximizes F1 score over a grid of thresholds on labeled data.",
            }
            save_summary()
        else:
            thr = float(np.percentile(scores, args.percentile))
            print(f"Flatness threshold (p{args.percentile}): {thr:.6f}")
            summary["flatness"] = {
                "threshold": float(thr),
                "percentile": float(args.percentile),
                "why_best": "Percentile of unlabeled metric distribution (assumes lower flatness indicates occupancy).",
            }
            save_summary()

    print(f"Saved threshold summary to: {args.output}")

    # Optional evaluation summary on labeled data
    if args.mode == "labeled" and labeled:
        metrics: dict[str, dict[str, float | list[int] | dict[str, int]]] = {
            "dataset_root": str(args.dataset_root),
            "label_distribution": label_dist if labeled else {},
        }

        def eval_detector(detector_name: str, scores: np.ndarray, thr: float, higher_is_positive: bool) -> None:
            preds = (scores > thr).astype(int) if higher_is_positive else (scores < thr).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            metrics[detector_name] = {
                "threshold": float(thr),
                "accuracy": float(np.mean(preds == labels)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "confusion_matrix": cm.tolist(),
            }

        if args.detector in {"energy", "both"} and "energy" in scores_cache:
            scores_e, labels = scores_cache["energy"]
            thr_e = summary.get("energy", {}).get("threshold")
            if isinstance(thr_e, (int, float)):
                eval_detector("energy", scores_e, float(thr_e), higher_is_positive=True)

        if args.detector in {"flatness", "both"} and "flatness" in scores_cache:
            scores_f, labels = scores_cache["flatness"]
            thr_f = summary.get("flatness", {}).get("threshold")
            if isinstance(thr_f, (int, float)):
                eval_detector("flatness", scores_f, float(thr_f), higher_is_positive=False)

        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics summary to: {args.metrics_output}")


if __name__ == "__main__":
    main()

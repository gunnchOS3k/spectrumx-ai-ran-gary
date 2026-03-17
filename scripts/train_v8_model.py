"""
Train v8 model: GradientBoosting ensemble with hardcoded tree export.

Key changes from v7:
- Back to 5M IQ samples (10M didn't help accuracy)
- GradientBoostingClassifier instead of LogisticRegression
- Export tree structures as arrays for pure-numpy inference
- Vectorized spectral kurtosis (kept from v7)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_IQ = 5_000_000


def load_iq(path: Path) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if np.iscomplexobj(data):
        n = min(MAX_IQ, data.shape[0])
        return np.array(data[:n], dtype=np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        n = min(MAX_IQ, data.shape[0])
        return (data[:n, 0].astype(np.float32) + 1j * data[:n, 1].astype(np.float32))
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        n = min(MAX_IQ * 2, data.shape[0])
        return (data[:n:2].astype(np.float32) + 1j * data[1:n:2].astype(np.float32))
    raise ValueError(f"Unsupported: shape={data.shape}, dtype={data.dtype}")


def extract_features(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    N = len(iq)
    mag = np.abs(iq)
    power = mag * mag
    mean_power = float(np.mean(power))

    nperseg = min(8192, N)
    _, psd = sig.welch(
        iq, fs=sample_rate, nperseg=nperseg,
        return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    psd_mean = float(np.mean(psd_mag))
    psd_std = float(np.std(psd_mag))
    psd_max = float(np.max(psd_mag))
    psd_min = float(np.min(psd_mag))

    p10, p25, p75, p90 = np.percentile(psd_mag, [10, 25, 75, 90])
    psd_iqr = float(p75 - p25)

    psd_pos = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    flatness = geo_mean / psd_mean if psd_mean > 0 else 0.0

    psd_sum = np.sum(psd_mag)
    if psd_sum > 0:
        psd_norm = psd_mag / psd_sum
        psd_norm_pos = psd_norm[psd_norm > 0]
        spectral_entropy = float(-np.sum(psd_norm_pos * np.log2(psd_norm_pos)))
    else:
        spectral_entropy = 0.0

    peak_to_mean = psd_max / (psd_mean + 1e-30)

    sk_seg_len = 1024
    n_segs = N // sk_seg_len
    if n_segs >= 8:
        iq_trimmed = iq[: n_segs * sk_seg_len].reshape(n_segs, sk_seg_len)
        window = np.hanning(sk_seg_len).astype(np.float32)
        fft_out = np.fft.fft(iq_trimmed * window[np.newaxis, :], axis=1)
        psd_segs = (np.abs(fft_out) ** 2) / (sk_seg_len * sample_rate)
        sk_per_bin = scipy_kurtosis(psd_segs, axis=0, fisher=True)
        spectral_kurt_mean = float(np.mean(sk_per_bin))
        spectral_kurt_max = float(np.max(sk_per_bin))
    else:
        spectral_kurt_mean = 0.0
        spectral_kurt_max = 0.0

    n_bins = len(psd_mag)
    quarter = n_bins // 4
    total_energy = float(np.sum(psd_mag)) + 1e-30
    band_energies = [
        float(np.sum(psd_mag[i * quarter: (i + 1) * quarter])) / total_energy
        for i in range(4)
    ]
    band_energy_std = float(np.std(band_energies))

    ac = float(np.abs(np.mean(iq[1:] * np.conj(iq[:-1])))) if N > 1 else 0.0

    mag_kurtosis = float(scipy_kurtosis(mag, fisher=True))
    mag_skewness = float(scipy_skew(mag))
    crest_factor = float(np.max(mag) / (np.sqrt(mean_power) + 1e-30))
    i_std = float(np.std(iq.real))
    q_std = float(np.std(iq.imag))
    iq_corr = float(np.abs(np.corrcoef(iq.real, iq.imag)[0, 1])) if N > 1 else 0.0
    power_var_ratio = float(np.var(power)) / (mean_power ** 2 + 1e-30)

    return np.array([
        psd_mean, psd_std, psd_max, psd_min,
        float(p10), float(p25), float(p75), float(p90), psd_iqr,
        flatness, spectral_entropy, peak_to_mean,
        spectral_kurt_mean, spectral_kurt_max,
        band_energies[0], band_energies[1], band_energies[2], band_energies[3],
        band_energy_std, ac,
        mean_power, float(np.std(power)), float(np.std(mag)),
        mag_kurtosis, mag_skewness, crest_factor,
        i_std, q_std, iq_corr, power_var_ratio,
    ], dtype=float)


FEATURE_NAMES = [
    "psd_mean", "psd_std", "psd_max", "psd_min", "psd_p10", "psd_p25",
    "psd_p75", "psd_p90", "psd_iqr", "flatness", "spectral_entropy",
    "peak_to_mean", "spectral_kurt_mean", "spectral_kurt_max",
    "band_e0", "band_e1", "band_e2", "band_e3", "band_energy_std",
    "autocorr", "mean_power", "std_power", "mag_std", "mag_kurtosis",
    "mag_skewness", "crest_factor", "i_std", "q_std", "iq_corr",
    "power_var_ratio",
]


def load_labeled(dataset_root: Path) -> tuple[list[np.ndarray], np.ndarray]:
    files_root = dataset_root / "files"
    user_dirs = [p for p in files_root.iterdir() if p.is_dir()]
    user_dir = user_dirs[0]
    labeled_dir = user_dir / "VLA_brutal"
    training_dir = user_dir / "trainingData"
    gt_path = labeled_dir / "groundtruth.csv"

    gt = pd.read_csv(gt_path)
    X, y = [], []
    for _, row in gt.iterrows():
        fname = row["filename"]
        fp = training_dir / fname
        if not fp.exists():
            fp = labeled_dir / fname
        if not fp.exists():
            continue
        X.append(load_iq(fp))
        y.append(int(row["label"]))
    return X, np.array(y)


def export_tree(tree) -> dict:
    """Extract tree structure as plain arrays."""
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value.reshape(-1).tolist(),
    }


def sweep_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    best_acc, best_thr = -1.0, 0.5
    for thr in np.linspace(0.01, 0.99, 2000):
        acc = accuracy_score(y_true, (probs >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
    )
    parser.add_argument("--sample-rate", type=float, default=1e6)
    parser.add_argument("--output", type=Path, default=Path("results/v8_model.json"))
    args = parser.parse_args()

    print(f"Loading labeled data ({MAX_IQ // 1_000_000}M IQ samples)...")
    X_iq, y = load_labeled(args.dataset_root)
    print(f"Labeled: {len(X_iq)} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    print(f"Extracting {len(FEATURE_NAMES)} features...")
    t0 = time.perf_counter()
    X_raw = np.array([extract_features(iq, args.sample_rate) for iq in X_iq])
    elapsed = time.perf_counter() - t0
    print(f"Feature extraction: {elapsed:.1f}s total, {elapsed / len(X_iq):.2f}s per file")
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # GradientBoosting — captures non-linear patterns
    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    print(f"Running 5-fold CV with GradientBoosting ({gbm_params['n_estimators']} trees)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        GradientBoostingClassifier(**gbm_params),
        X_scaled, y, cv=cv, method="predict_proba",
    )[:, 1]

    # Balanced evaluation
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_bal = min(len(idx_pos), len(idx_neg))
    rng = np.random.default_rng(42)
    idx_bal = np.concatenate([rng.choice(idx_pos, n_bal, replace=False),
                              rng.choice(idx_neg, n_bal, replace=False)])
    thr_bal, acc_bal = sweep_threshold(cv_probs[idx_bal], y[idx_bal])
    print(f"Balanced CV accuracy: {acc_bal:.4f} at threshold {thr_bal:.4f}")

    preds_bal = (cv_probs[idx_bal] >= thr_bal).astype(int)
    cm = confusion_matrix(y[idx_bal], preds_bal, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    print(f"  TP={tp / total:.2f} FP={fp / total:.2f} FN={fn / total:.2f} TN={tn / total:.2f}")

    # Train final model
    print("Training final GBM model...")
    model = GradientBoostingClassifier(**gbm_params)
    model.fit(X_scaled, y)

    final_probs = model.predict_proba(X_scaled)[:, 1]
    thr_final, acc_final = sweep_threshold(final_probs[idx_bal], y[idx_bal])
    print(f"Final balanced accuracy: {acc_final:.4f} at threshold {thr_final:.4f}")

    preds_final = (final_probs[idx_bal] >= thr_final).astype(int)
    cm2 = confusion_matrix(y[idx_bal], preds_final, labels=[0, 1])
    tn2, fp2, fn2, tp2 = cm2.ravel()
    total2 = tn2 + fp2 + fn2 + tp2
    print(f"  TP={tp2 / total2:.2f} FP={fp2 / total2:.2f} FN={fn2 / total2:.2f} TN={tn2 / total2:.2f}")

    # Feature importance
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for name, imp in ranked[:10]:
        print(f"  {name:25s} {imp:.4f}")

    # Export trees
    trees = []
    for estimators_at_stage in model.estimators_:
        tree = estimators_at_stage[0]
        trees.append(export_tree(tree))

    print(f"\nExported {len(trees)} trees")
    total_nodes = sum(len(t["feature"]) for t in trees)
    print(f"Total nodes: {total_nodes}")

    payload = {
        "sample_rate": float(args.sample_rate),
        "max_iq_samples": MAX_IQ,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "gbm_learning_rate": model.learning_rate,
        "gbm_init_value": float(model.init_.class_prior_[1]),
        "trees": trees,
        "optimal_threshold": float(thr_final),
        "balanced_cv_accuracy": float(acc_bal),
        "final_balanced_accuracy": float(acc_final),
        "extraction_time_per_file_s": elapsed / len(X_iq),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved v8 model to: {args.output}")

    # Print summary (not full JSON — it's huge with all the trees)
    summary = {k: v for k, v in payload.items() if k != "trees"}
    summary["n_trees"] = len(trees)
    summary["total_tree_nodes"] = total_nodes
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

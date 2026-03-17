"""
Train v11 model: GBM with asymmetric sample weights penalizing FP.

Key insight from v9/v10: threshold tuning is zero-sum (reducing FP loses TP).
We need the model itself to learn a sharper boundary around noise.

Strategy:
- Weight noise samples 3x during training (penalizes false positives)
- Prior correction (INIT_LOG_ODDS=0) for 50/50 test distribution
- Threshold sweep on balanced CV
- Same 30 features, 5M IQ, vectorized kurtosis
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
    parser.add_argument("--output", type=Path, default=Path("results/v11_model.json"))
    parser.add_argument("--noise-weight", type=float, default=3.0,
                        help="Weight multiplier for noise (class 0) samples")
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

    # Asymmetric sample weights: penalize FP by weighting noise higher
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[y == 0] = args.noise_weight
    print(f"Sample weights: signal=1.0, noise={args.noise_weight}")

    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    # 5-fold CV with sample weights
    print(f"Running 5-fold CV with GBM (noise_weight={args.noise_weight})...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        gbm = GradientBoostingClassifier(**gbm_params)
        gbm.fit(X_scaled[train_idx], y[train_idx],
                sample_weight=sample_weight[train_idx])
        cv_probs[val_idx] = gbm.predict_proba(X_scaled[val_idx])[:, 1]

    # Apply prior correction: shift from 2:1 to 1:1
    # GBM outputs raw score = init + tree_sum
    # init = log(p/(1-p)) where p=2/3 → log(2)≈0.693
    # For 50/50 test: corrected init = 0
    # prob_corrected = sigmoid(logit(prob_orig) - log(2))
    log2 = math.log(2.0)
    cv_logits = np.log(cv_probs / (1 - cv_probs + 1e-15) + 1e-15)
    cv_logits_corrected = cv_logits - log2
    cv_probs_corrected = 1.0 / (1.0 + np.exp(-cv_logits_corrected))

    # Balanced evaluation
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_bal = min(len(idx_pos), len(idx_neg))
    rng = np.random.default_rng(42)
    idx_bal = np.concatenate([rng.choice(idx_pos, n_bal, replace=False),
                              rng.choice(idx_neg, n_bal, replace=False)])

    thr_bal, acc_bal = sweep_threshold(cv_probs_corrected[idx_bal], y[idx_bal])
    print(f"Balanced CV accuracy (prior-corrected): {acc_bal:.4f} at threshold {thr_bal:.4f}")

    preds_bal = (cv_probs_corrected[idx_bal] >= thr_bal).astype(int)
    cm = confusion_matrix(y[idx_bal], preds_bal, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    print(f"  TP={tp / total:.2f} FP={fp / total:.2f} FN={fn / total:.2f} TN={tn / total:.2f}")

    # Train final model
    print("Training final GBM model...")
    model = GradientBoostingClassifier(**gbm_params)
    model.fit(X_scaled, y, sample_weight=sample_weight)

    # Feature importance
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for name, imp in ranked[:10]:
        print(f"  {name:25s} {imp:.4f}")

    # Export trees
    trees = []
    for estimators_at_stage in model.estimators_:
        trees.append(export_tree(estimators_at_stage[0]))

    total_nodes = sum(len(t["feature"]) for t in trees)
    print(f"\nExported {len(trees)} trees, {total_nodes} total nodes")

    init_prior = float(model.init_.class_prior_[1])
    init_log_odds = math.log(init_prior / (1 - init_prior))
    corrected_init = init_log_odds - log2
    print(f"Original init_log_odds: {init_log_odds:.4f}")
    print(f"Corrected for 50/50: {corrected_init:.4f}")

    payload = {
        "sample_rate": float(args.sample_rate),
        "max_iq_samples": MAX_IQ,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "gbm_learning_rate": model.learning_rate,
        "gbm_init_log_odds_original": init_log_odds,
        "gbm_init_log_odds_corrected": corrected_init,
        "trees": trees,
        "optimal_threshold": float(thr_bal),
        "balanced_cv_accuracy": float(acc_bal),
        "noise_weight": args.noise_weight,
        "extraction_time_per_file_s": elapsed / len(X_iq),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved v11 model to: {args.output}")

    summary = {k: v for k, v in payload.items() if k != "trees"}
    summary["n_trees"] = len(trees)
    summary["total_tree_nodes"] = total_nodes
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

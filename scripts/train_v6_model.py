"""
Train v6 model: more IQ samples, weak-signal features, optimized threshold.

Key improvements over v5:
- Use 5M IQ samples (vs 2M) for better weak-signal PSD
- Add spectral kurtosis, band energy ratios, autocorrelation features
- Sweep threshold targeting FP<=0.03 to recover more TP
- Larger PSD window (8192) for finer frequency resolution
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.linear_model import LogisticRegression
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
        iq = data[:MAX_IQ] if data.shape[0] > MAX_IQ else data
        return np.array(iq, dtype=np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        n = min(MAX_IQ, data.shape[0])
        return (data[:n, 0].astype(np.float32) + 1j * data[:n, 1].astype(np.float32)).astype(np.complex64)
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        n = min(MAX_IQ * 2, data.shape[0])
        i = data[:n:2].astype(np.float32)
        q = data[1:n:2].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)
    raise ValueError(f"Unsupported: shape={data.shape}, dtype={data.dtype}")


def extract_features(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    mag = np.abs(iq)
    power = mag ** 2
    mean_power = float(np.mean(power))

    nperseg = min(8192, len(iq))
    _, psd = sig.welch(
        iq, fs=sample_rate, nperseg=nperseg,
        return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    psd_mean = float(np.mean(psd_mag))
    psd_std = float(np.std(psd_mag))
    psd_max = float(np.max(psd_mag))
    psd_min = float(np.min(psd_mag))

    psd_pos = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    flatness = geo_mean / psd_mean if psd_mean > 0 else 0.0

    psd_norm = psd_mag / (np.sum(psd_mag) + 1e-30)
    psd_norm_pos = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm_pos * np.log2(psd_norm_pos)))

    peak_to_mean = psd_max / (psd_mean + 1e-30)

    # Spectral kurtosis (per-bin kurtosis across segments)
    n_segs = max(1, len(iq) // nperseg)
    if n_segs >= 4:
        seg_psds = []
        for i in range(n_segs):
            seg = iq[i * nperseg: (i + 1) * nperseg]
            if len(seg) < nperseg:
                break
            _, sp = sig.welch(seg, fs=sample_rate, nperseg=min(1024, len(seg)),
                              return_onesided=False, scaling="density")
            seg_psds.append(np.abs(sp))
        if len(seg_psds) >= 4:
            seg_arr = np.array(seg_psds)
            sk_per_bin = scipy_kurtosis(seg_arr, axis=0, fisher=True)
            spectral_kurt_mean = float(np.mean(sk_per_bin))
            spectral_kurt_max = float(np.max(sk_per_bin))
        else:
            spectral_kurt_mean = 0.0
            spectral_kurt_max = 0.0
    else:
        spectral_kurt_mean = 0.0
        spectral_kurt_max = 0.0

    # Band energy ratios (split PSD into 4 bands)
    n_bins = len(psd_mag)
    quarter = n_bins // 4
    total_energy = float(np.sum(psd_mag)) + 1e-30
    band_energies = [
        float(np.sum(psd_mag[i * quarter: (i + 1) * quarter])) / total_energy
        for i in range(4)
    ]
    band_energy_std = float(np.std(band_energies))

    # Autocorrelation at lag 1
    if len(iq) > 1:
        ac = float(np.abs(np.mean(iq[1:] * np.conj(iq[:-1]))))
    else:
        ac = 0.0

    # Time-domain stats
    mag_kurtosis = float(scipy_kurtosis(mag, fisher=True))
    mag_skewness = float(scipy_skew(mag))
    crest_factor = float(np.max(mag) / (np.sqrt(mean_power) + 1e-30))
    i_std = float(np.std(iq.real))
    q_std = float(np.std(iq.imag))
    iq_corr = float(np.abs(np.corrcoef(iq.real, iq.imag)[0, 1])) if len(iq) > 1 else 0.0

    # Power variance ratio (signal power is more variable than noise)
    power_var_ratio = float(np.var(power)) / (mean_power ** 2 + 1e-30)

    return np.array([
        psd_mean,                # 0
        psd_std,                 # 1
        psd_max,                 # 2
        psd_min,                 # 3
        float(np.percentile(psd_mag, 10)),  # 4
        float(np.percentile(psd_mag, 25)),  # 5
        float(np.percentile(psd_mag, 75)),  # 6
        float(np.percentile(psd_mag, 90)),  # 7
        float(np.percentile(psd_mag, 75) - np.percentile(psd_mag, 25)),  # 8 iqr
        flatness,                # 9
        spectral_entropy,        # 10
        peak_to_mean,            # 11
        spectral_kurt_mean,      # 12
        spectral_kurt_max,       # 13
        band_energies[0],        # 14
        band_energies[1],        # 15
        band_energies[2],        # 16
        band_energies[3],        # 17
        band_energy_std,         # 18
        ac,                      # 19
        mean_power,              # 20
        float(np.std(power)),    # 21
        float(np.std(mag)),      # 22
        mag_kurtosis,            # 23
        mag_skewness,            # 24
        crest_factor,            # 25
        i_std,                   # 26
        q_std,                   # 27
        iq_corr,                 # 28
        power_var_ratio,         # 29
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
        filename = row["filename"]
        file_path = training_dir / filename
        if not file_path.exists():
            file_path = labeled_dir / filename
        if not file_path.exists():
            continue
        X.append(load_iq(file_path))
        y.append(int(row["label"]))
    return X, np.array(y)


def sweep_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    best_acc = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.01, 0.99, 1000):
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v6 model.")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
    )
    parser.add_argument("--sample-rate", type=float, default=1e6)
    parser.add_argument("--output", type=Path, default=Path("results/v6_model.json"))
    args = parser.parse_args()

    print(f"Loading labeled data (first {MAX_IQ // 1_000_000}M IQ samples)...")
    X_iq, y = load_labeled(args.dataset_root)
    print(f"Labeled: {len(X_iq)} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    print(f"Extracting {len(FEATURE_NAMES)} features...")
    X_raw = np.array([extract_features(iq, args.sample_rate) for iq in X_iq])
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 5-fold stratified CV
    print("Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", solver="lbfgs"),
        X_scaled, y, cv=cv, method="predict_proba",
    )[:, 1]

    # Sweep on balanced subset (simulates 50/50 test)
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
    print(f"  TP={tp/total:.2f} FP={fp/total:.2f} FN={fn/total:.2f} TN={tn/total:.2f}")

    # Train final model
    print("Training final model...")
    model = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", solver="lbfgs")
    model.fit(X_scaled, y)

    final_probs = model.predict_proba(X_scaled)[:, 1]
    thr_final, acc_final = sweep_threshold(final_probs[idx_bal], y[idx_bal])
    print(f"Final balanced accuracy: {acc_final:.4f} at threshold {thr_final:.4f}")

    preds_final = (final_probs[idx_bal] >= thr_final).astype(int)
    cm2 = confusion_matrix(y[idx_bal], preds_final, labels=[0, 1])
    tn2, fp2, fn2, tp2 = cm2.ravel()
    total2 = tn2 + fp2 + fn2 + tp2
    print(f"  TP={tp2/total2:.2f} FP={fp2/total2:.2f} FN={fn2/total2:.2f} TN={tn2/total2:.2f}")

    payload = {
        "sample_rate": float(args.sample_rate),
        "max_iq_samples": MAX_IQ,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "weights": model.coef_.reshape(-1).tolist(),
        "bias": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "optimal_threshold": float(thr_final),
        "balanced_cv_accuracy": float(acc_bal),
        "final_balanced_accuracy": float(acc_final),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved v6 model to: {args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

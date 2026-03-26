"""
Train v14 model: 33 features targeting the exact FP failure mode of v9.

Diagnosis of v9 (acc=0.85, FP=0.04):
- The GBM relies 77% on psd_max (raw spectral maximum).
- 8 noise files have a random high-power spike -> high psd_max -> FP.
- These spikes are NOT real signals: they are narrow, fleeting, and
  sit close to the local noise floor.

Three new features specifically counter this:
  31. psd_top10_frac   : fraction of total PSD energy in top-10 bins.
                         Real narrowband signals score HIGH.
                         Flat noise scores LOW (~10/N_bins).
  32. psd_max_to_p10   : psd_max / psd_p10 (peak above noise floor).
                         Real peaks stand FAR above the 10th-percentile floor.
                         Noise spikes are only modestly above it.
  33. peak_bin_cv      : CV of argmax(PSD) across 8 time windows.
                         Real signals stay at the SAME frequency (low CV).
                         Noise peaks wander randomly (high CV).

Target on leaderboard: FP=0.00, TP>=0.37, acc>=0.87 (matching sub_test3).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

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

    # --- Global PSD via Welch ---
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

    psd_sum = float(np.sum(psd_mag))
    if psd_sum > 0:
        psd_norm = psd_mag / psd_sum
        psd_norm_pos = psd_norm[psd_norm > 0]
        spectral_entropy = float(-np.sum(psd_norm_pos * np.log2(psd_norm_pos)))
    else:
        spectral_entropy = 0.0

    peak_to_mean = psd_max / (psd_mean + 1e-30)

    # --- Vectorised spectral kurtosis (from v7+) ---
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

    # --- Band energies ---
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

    # ---- NEW FEATURES targeting FP failure mode ----

    # 31. psd_top10_frac: fraction of total PSD energy in the top-10 bins.
    # Real narrowband signals concentrate energy; flat noise does not.
    k_top = min(10, len(psd_mag))
    top_k_sum = float(np.sum(np.partition(psd_mag, -k_top)[-k_top:]))
    psd_top10_frac = top_k_sum / (psd_sum + 1e-30)

    # 32. psd_max_to_p10: peak vs noise-floor proxy.
    # A genuine spectral peak towers above the 10th percentile.
    # A random noise spike is only modestly elevated.
    psd_max_to_p10 = psd_max / (float(p10) + 1e-30)

    # 33. peak_bin_cv: coefficient of variation of peak-frequency location
    # across 8 time windows (lower = more consistent = more signal-like).
    win_len_cv = min(524288, N // 8)
    n_cv_wins = N // win_len_cv if win_len_cv >= 1024 else 0
    if n_cv_wins >= 4:
        win_cv = np.hanning(win_len_cv).astype(np.float32)
        peak_bins: list[float] = []
        for w in range(n_cv_wins):
            seg = iq[w * win_len_cv: (w + 1) * win_len_cv]
            X = np.fft.fft(seg * win_cv)
            psd_w = (np.abs(X) ** 2)
            peak_bins.append(float(np.argmax(psd_w)))
        pb = np.array(peak_bins, dtype=float)
        pb_mean = float(np.mean(pb))
        peak_bin_cv = float(np.std(pb)) / (pb_mean + 1.0)
    else:
        # Undetermined - use neutral value (moderate CV)
        peak_bin_cv = 0.5

    return np.array([
        # v9's 30 features (unchanged)
        psd_mean, psd_std, psd_max, psd_min,
        float(p10), float(p25), float(p75), float(p90), psd_iqr,
        flatness, spectral_entropy, peak_to_mean,
        spectral_kurt_mean, spectral_kurt_max,
        band_energies[0], band_energies[1], band_energies[2], band_energies[3],
        band_energy_std, ac,
        mean_power, float(np.std(power)), float(np.std(mag)),
        mag_kurtosis, mag_skewness, crest_factor,
        i_std, q_std, iq_corr, power_var_ratio,
        # 3 new features
        psd_top10_frac,     # 30
        psd_max_to_p10,     # 31
        peak_bin_cv,        # 32
    ], dtype=float)


FEATURE_NAMES = [
    "psd_mean", "psd_std", "psd_max", "psd_min", "psd_p10", "psd_p25",
    "psd_p75", "psd_p90", "psd_iqr", "flatness", "spectral_entropy",
    "peak_to_mean", "spectral_kurt_mean", "spectral_kurt_max",
    "band_e0", "band_e1", "band_e2", "band_e3", "band_energy_std",
    "autocorr", "mean_power", "std_power", "mag_std", "mag_kurtosis",
    "mag_skewness", "crest_factor", "i_std", "q_std", "iq_corr",
    "power_var_ratio",
    # new
    "psd_top10_frac", "psd_max_to_p10", "peak_bin_cv",
]


def load_labeled(dataset_root: Path):
    files_root = dataset_root / "files"
    user_dir = next(p for p in files_root.iterdir() if p.is_dir())
    labeled_dir = user_dir / "VLA_brutal"
    training_dir = user_dir / "trainingData"
    gt = pd.read_csv(labeled_dir / "groundtruth.csv")
    X, y = [], []
    for _, row in gt.iterrows():
        fp = training_dir / row["filename"]
        if not fp.exists():
            fp = labeled_dir / row["filename"]
        if not fp.exists():
            continue
        X.append(load_iq(fp))
        y.append(int(row["label"]))
    return X, np.array(y)


def export_tree(tree):
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value.reshape(-1).tolist(),
    }


def sweep_threshold(probs, y_true):
    best_acc, best_thr = -1.0, 0.5
    for thr in np.linspace(0.01, 0.99, 2000):
        acc = accuracy_score(y_true, (probs >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def main():
    dataset_root = Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset")
    output = Path("results/v14_model.json")
    sample_rate = 1e6

    print(f"Loading labeled data ({MAX_IQ // 1_000_000}M IQ samples)...")
    X_iq, y = load_labeled(dataset_root)
    print(f"Labeled: {len(X_iq)} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    print(f"Extracting {len(FEATURE_NAMES)} features...")
    t0 = time.perf_counter()
    X_raw = np.array([extract_features(iq, sample_rate) for iq in X_iq])
    elapsed = time.perf_counter() - t0
    print(f"Feature extraction: {elapsed:.1f}s total, {elapsed / len(X_iq):.2f}s/file")
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Show new feature distributions by class - diagnostic
    for fi, fname in enumerate(FEATURE_NAMES[-3:], start=30):
        v0 = X_raw[y == 0, fi]
        v1 = X_raw[y == 1, fi]
        print(f"  {fname}: noise median={np.median(v0):.4f} signal median={np.median(v1):.4f}")

    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    # 5-fold CV
    print("Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = np.zeros(len(y))
    for fold, (tr, val) in enumerate(cv.split(X_scaled, y)):
        gbm = GradientBoostingClassifier(**gbm_params)
        gbm.fit(X_scaled[tr], y[tr])
        cv_probs[val] = gbm.predict_proba(X_scaled[val])[:, 1]

    # Apply prior correction (train=2:1, test=1:1)
    log2 = math.log(2.0)
    cv_logits = np.log(np.clip(cv_probs, 1e-7, 1 - 1e-7) / (1 - np.clip(cv_probs, 1e-7, 1 - 1e-7)))
    cv_logits_corr = cv_logits - log2
    cv_probs_corr = 1.0 / (1.0 + np.exp(-cv_logits_corr))

    # Balanced subset evaluation
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_bal = min(len(idx_pos), len(idx_neg))
    rng = np.random.default_rng(42)
    idx_bal = np.concatenate([rng.choice(idx_pos, n_bal, replace=False),
                              rng.choice(idx_neg, n_bal, replace=False)])
    y_bal = y[idx_bal]
    p_bal = cv_probs_corr[idx_bal]

    thr, acc = sweep_threshold(p_bal, y_bal)
    preds = (p_bal >= thr).astype(int)
    cm = confusion_matrix(y_bal, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    print(f"Balanced CV accuracy: {acc:.4f} at threshold={thr:.4f}")
    print(f"  TP={tp/total:.2f} FP={fp/total:.2f} FN={fn/total:.2f} TN={tn/total:.2f}")

    # Also try FP=0 operating point
    print("\nSweeping for FP-constrained thresholds:")
    for fp_max in [0.00, 0.01, 0.02]:
        for thr_try in np.linspace(0.01, 0.99, 2000):
            preds_try = (p_bal >= thr_try).astype(int)
            cm_try = confusion_matrix(y_bal, preds_try, labels=[0, 1])
            tn_t, fp_t, fn_t, tp_t = cm_try.ravel()
            tot = tn_t + fp_t + fn_t + tp_t
            fp_rate = fp_t / tot
            if fp_rate <= fp_max:
                tp_rate = tp_t / tot
                acc_try = accuracy_score(y_bal, preds_try)
                print(f"  FP<={fp_max:.2f}: thr={thr_try:.4f}  acc={acc_try:.4f}  "
                      f"TP={tp_rate:.2f} FP={fp_rate:.2f}")
                break

    # Train final model
    print("\nTraining final model...")
    model = GradientBoostingClassifier(**gbm_params)
    model.fit(X_scaled, y)

    # Feature importance
    ranked = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for name, imp in ranked[:10]:
        print(f"  {name:25s} {imp:.4f}")

    # Predict + prior-correct on full set, find final threshold
    full_probs_raw = model.predict_proba(X_scaled)[:, 1]
    full_logits = np.log(np.clip(full_probs_raw, 1e-7, 1-1e-7) / (1 - np.clip(full_probs_raw, 1e-7, 1-1e-7)))
    full_probs_corr = 1.0 / (1.0 + np.exp(-(full_logits - log2)))

    p_bal_final = full_probs_corr[idx_bal]
    thr_final, acc_final = sweep_threshold(p_bal_final, y_bal)
    preds_final = (p_bal_final >= thr_final).astype(int)
    cm_f = confusion_matrix(y_bal, preds_final, labels=[0, 1])
    tn_f, fp_f, fn_f, tp_f = cm_f.ravel()
    total_f = tn_f + fp_f + fn_f + tp_f
    print(f"\nFinal balanced accuracy: {acc_final:.4f} at threshold={thr_final:.4f}")
    print(f"  TP={tp_f/total_f:.2f} FP={fp_f/total_f:.2f} FN={fn_f/total_f:.2f} TN={tn_f/total_f:.2f}")

    init_prior = float(model.init_.class_prior_[1])
    init_log_odds = math.log(init_prior / (1 - init_prior))

    trees = [export_tree(est[0]) for est in model.estimators_]
    total_nodes = sum(len(t["feature"]) for t in trees)
    print(f"\n{len(trees)} trees, {total_nodes} total nodes")

    payload = {
        "sample_rate": float(sample_rate),
        "max_iq_samples": MAX_IQ,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "gbm_learning_rate": model.learning_rate,
        "gbm_init_log_odds": init_log_odds,
        "gbm_init_log_odds_corrected": init_log_odds - log2,
        "trees": trees,
        "optimal_threshold": float(thr_final),
        "balanced_cv_accuracy": float(acc),
        "final_balanced_accuracy": float(acc_final),
        "extraction_time_per_file_s": elapsed / len(X_iq),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    summary = {k: v for k, v in payload.items() if k != "trees"}
    summary["n_trees"] = len(trees)
    summary["total_nodes"] = total_nodes
    print(f"\nSaved to: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

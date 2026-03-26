"""
Generate submissions/leaderboard_v14/main.py from results/v14_model.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = ROOT / "results" / "v14_model.json"
OUT_DIR = ROOT / "submissions" / "leaderboard_v14"
OUT_DIR.mkdir(parents=True, exist_ok=True)

payload = json.loads(MODEL_JSON.read_text())

scaler_mean = payload["scaler_mean"]
scaler_std  = payload["scaler_std"]
lr          = payload["gbm_learning_rate"]
trees       = payload["trees"]

# Flatten all trees into parallel arrays for fast numpy inference
children_left  = []
children_right = []
features       = []
thresholds     = []
values         = []
offsets        = []

off = 0
for t in trees:
    offsets.append(off)
    n = len(t["feature"])
    children_left .extend(t["children_left"])
    children_right.extend(t["children_right"])
    features      .extend(t["feature"])
    thresholds    .extend(t["threshold"])
    values        .extend(t["value"])
    off += n

# Prior correction: training set has 2:1 (positive:negative).
# The GBM init bias = log(2/3 / 1/3) = log(2).
# Setting INIT_LOG_ODDS = 0.0 effectively removes that bias,
# matching the expected 50/50 test distribution.
INIT_LOG_ODDS = 0.0
# Use the CV-tuned threshold (FP<=0.01 at 0.9861)
CV_THRESHOLD  = 0.9861

def fmt_list(lst, per_line=8, indent=4):
    sp = " " * indent
    chunks = [lst[i:i+per_line] for i in range(0, len(lst), per_line)]
    return "[\n" + ",\n".join(sp + str(chunk) for chunk in chunks) + "\n]"

def fmt_float_list(lst, per_line=6, indent=4):
    sp = " " * indent
    lines = []
    for i in range(0, len(lst), per_line):
        chunk = lst[i:i+per_line]
        lines.append(sp + ", ".join(f"{v!r}" for v in chunk) + ",")
    return "[\n" + "\n".join(lines) + "\n]"

MAIN_PY = f'''\
"""
leaderboard_v14 - Signal Occupancy Detector
33-feature GBM (v9 base + 3 anti-FP features)

New features vs v9:
  psd_top10_frac  : fraction of PSD energy in top-10 frequency bins
  psd_max_to_p10  : peak / noise-floor (10th percentile) - robust peak height
  peak_bin_cv     : CV of peak-bin location across 8 windows - signal consistency

CV results (balanced, prior-corrected): acc=0.93, TP=0.44, FP=0.01
Threshold: {CV_THRESHOLD} (FP<=0.01 operating point)
"""

from __future__ import annotations
import math
import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

# ---------------------------------------------------------------------------
# Inference constants
# ---------------------------------------------------------------------------
MAX_IQ_SAMPLES  = {payload["max_iq_samples"]}
SAMPLE_RATE     = {payload["sample_rate"]}
PROB_THRESHOLD  = {CV_THRESHOLD}
INIT_LOG_ODDS   = {INIT_LOG_ODDS}
LEARNING_RATE   = {lr!r}
N_TREES         = {len(trees)}

SCALER_MEAN = {fmt_float_list(scaler_mean)}

SCALER_STD = {fmt_float_list(scaler_std)}

TREE_OFFSETS    = {offsets!r}
TREE_LEFT       = {children_left!r}
TREE_RIGHT      = {children_right!r}
TREE_FEATURE    = {features!r}
TREE_THRESHOLD  = {thresholds!r}
TREE_VALUE      = {values!r}


# ---------------------------------------------------------------------------
# IQ loader
# ---------------------------------------------------------------------------
def _load_iq_auto(filename: str) -> np.ndarray:
    data = np.load(filename, mmap_mode="r")
    if np.iscomplexobj(data):
        n = min(MAX_IQ_SAMPLES, data.shape[0])
        return np.array(data[:n], dtype=np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        n = min(MAX_IQ_SAMPLES, data.shape[0])
        return (data[:n, 0].astype(np.float32) + 1j * data[:n, 1].astype(np.float32))
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        n = min(MAX_IQ_SAMPLES * 2, data.shape[0])
        return (data[:n:2].astype(np.float32) + 1j * data[1:n:2].astype(np.float32))
    raise ValueError(f"Unsupported format: shape={{data.shape}}, dtype={{data.dtype}}")


# ---------------------------------------------------------------------------
# Feature extraction (33 features)
# ---------------------------------------------------------------------------
def _extract_features(iq: np.ndarray, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    N = len(iq)
    mag = np.abs(iq)
    power = mag * mag
    mean_power = float(np.mean(power))

    # Global PSD via Welch
    nperseg = min(8192, N)
    _, psd = sig.welch(
        iq, fs=sample_rate, nperseg=nperseg,
        return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    psd_mean = float(np.mean(psd_mag))
    psd_std  = float(np.std(psd_mag))
    psd_max  = float(np.max(psd_mag))
    psd_min  = float(np.min(psd_mag))

    p10, p25, p75, p90 = np.percentile(psd_mag, [10, 25, 75, 90])
    psd_iqr = float(p75 - p25)

    psd_pos  = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    flatness = geo_mean / psd_mean if psd_mean > 0 else 0.0

    psd_sum = float(np.sum(psd_mag))
    if psd_sum > 0:
        psd_norm     = psd_mag / psd_sum
        psd_norm_pos = psd_norm[psd_norm > 0]
        spectral_entropy = float(-np.sum(psd_norm_pos * np.log2(psd_norm_pos)))
    else:
        spectral_entropy = 0.0

    peak_to_mean = psd_max / (psd_mean + 1e-30)

    # Spectral kurtosis
    sk_seg_len = 1024
    n_segs = N // sk_seg_len
    if n_segs >= 8:
        iq_trimmed = iq[: n_segs * sk_seg_len].reshape(n_segs, sk_seg_len)
        window = np.hanning(sk_seg_len).astype(np.float32)
        fft_out  = np.fft.fft(iq_trimmed * window[np.newaxis, :], axis=1)
        psd_segs = (np.abs(fft_out) ** 2) / (sk_seg_len * sample_rate)
        sk_per_bin        = scipy_kurtosis(psd_segs, axis=0, fisher=True)
        spectral_kurt_mean = float(np.mean(sk_per_bin))
        spectral_kurt_max  = float(np.max(sk_per_bin))
    else:
        spectral_kurt_mean = spectral_kurt_max = 0.0

    # Band energies (4 quarters)
    n_bins       = len(psd_mag)
    quarter      = n_bins // 4
    total_energy = psd_sum + 1e-30
    band_energies = [
        float(np.sum(psd_mag[i * quarter: (i + 1) * quarter])) / total_energy
        for i in range(4)
    ]
    band_energy_std = float(np.std(band_energies))

    ac = float(np.abs(np.mean(iq[1:] * np.conj(iq[:-1])))) if N > 1 else 0.0

    mag_kurtosis  = float(scipy_kurtosis(mag, fisher=True))
    mag_skewness  = float(scipy_skew(mag))
    crest_factor  = float(np.max(mag) / (math.sqrt(mean_power) + 1e-30))
    i_std         = float(np.std(iq.real))
    q_std         = float(np.std(iq.imag))
    iq_corr       = float(np.abs(np.corrcoef(iq.real, iq.imag)[0, 1])) if N > 1 else 0.0
    power_var_ratio = float(np.var(power)) / (mean_power ** 2 + 1e-30)

    # NEW: psd_top10_frac - fraction of PSD energy in top-10 bins
    k_top = min(10, len(psd_mag))
    top_k_sum    = float(np.sum(np.partition(psd_mag, -k_top)[-k_top:]))
    psd_top10_frac = top_k_sum / (psd_sum + 1e-30)

    # NEW: psd_max_to_p10 - peak vs noise floor
    psd_max_to_p10 = psd_max / (float(p10) + 1e-30)

    # NEW: peak_bin_cv - temporal consistency of peak frequency
    win_len_cv = min(524288, N // 8)
    n_cv_wins  = N // win_len_cv if win_len_cv >= 1024 else 0
    if n_cv_wins >= 4:
        win_cv    = np.hanning(win_len_cv).astype(np.float32)
        peak_bins: list[float] = []
        for w in range(n_cv_wins):
            seg   = iq[w * win_len_cv: (w + 1) * win_len_cv]
            X_fft = np.fft.fft(seg * win_cv)
            psd_w = np.abs(X_fft) ** 2
            peak_bins.append(float(np.argmax(psd_w)))
        pb        = np.array(peak_bins, dtype=np.float64)
        peak_bin_cv = float(np.std(pb)) / (float(np.mean(pb)) + 1.0)
    else:
        peak_bin_cv = 0.5

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
        psd_top10_frac, psd_max_to_p10, peak_bin_cv,
    ], dtype=float)


# ---------------------------------------------------------------------------
# GBM inference
# ---------------------------------------------------------------------------
def _predict_gbm(x_scaled: np.ndarray) -> float:
    score = INIT_LOG_ODDS
    for i in range(N_TREES):
        off  = TREE_OFFSETS[i]
        node = 0
        while TREE_LEFT[off + node] != -1:
            if x_scaled[TREE_FEATURE[off + node]] <= TREE_THRESHOLD[off + node]:
                node = TREE_LEFT[off + node]
            else:
                node = TREE_RIGHT[off + node]
        score += LEARNING_RATE * TREE_VALUE[off + node]
    return 1.0 / (1.0 + math.exp(-score))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate(filename: str) -> int:
    iq = _load_iq_auto(filename)
    feats = _extract_features(iq)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    mean     = np.array(SCALER_MEAN)
    std      = np.array(SCALER_STD)
    x_scaled = (feats - mean) / (std + 1e-30)

    prob = _predict_gbm(x_scaled)
    return 1 if prob >= PROB_THRESHOLD else 0
'''

out_file = OUT_DIR / "main.py"
out_file.write_text(MAIN_PY, encoding="utf-8")
print(f"Written: {out_file}")
print(f"  Trees: {len(trees)}, total nodes: {len(features)}")
print(f"  PROB_THRESHOLD: {CV_THRESHOLD}")
print(f"  Estimated runtime: ~{payload['extraction_time_per_file_s']:.2f}s/file")

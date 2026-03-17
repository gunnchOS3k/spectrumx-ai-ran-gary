"""
v7 submission: optimized PSD + LogReg with 30 features, 10M IQ samples.

Improvements over v6:
- Vectorized spectral kurtosis via batched FFT (no per-segment Welch loop)
- Full 10M IQ samples (vs 5M) for better weak-signal detection
- Batched percentile computation
- Balanced CV accuracy: 93.75%, FP=0.00
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

MAX_IQ_SAMPLES = 10_000_000

LOGREG_WEIGHTS = [
    0.087522703678275,
    0.14909287388750006,
    0.2044360105564414,
    -0.07237384202304117,
    0.028559387067494172,
    0.025603384070987677,
    0.10394580805963113,
    0.1322169009106327,
    0.14181498400850925,
    -0.12426437898412786,
    -0.07412856293798353,
    0.11655625392962833,
    0.29372544251258564,
    0.3686767220999022,
    -0.8175545501130762,
    -0.7398559733987039,
    0.8612290863825676,
    0.7686150949151952,
    0.18269992199777366,
    0.05227059466864454,
    0.08760769066861293,
    0.08920207932028465,
    0.045817773182060983,
    0.0314327598968494,
    0.028053096836237714,
    0.15676703607455522,
    0.0454439795224821,
    0.04559252064194033,
    -0.12727250208448873,
    0.029410646393339974,
]
LOGREG_BIAS = 1.720838264730621

SCALER_MEAN = [
    1.0388139953336022e-08,
    3.7504697358743216e-09,
    1.9417661955752526e-07,
    3.0880497182509774e-09,
    8.911934041289092e-09,
    9.127058405583588e-09,
    1.1983426918132922e-08,
    1.289543949627621e-08,
    2.856368512549334e-09,
    0.974831716105869,
    12.941727681954701,
    20.88541951844191,
    6.818555257717768,
    65.50887124141057,
    0.23379017629636714,
    0.23310548676716683,
    0.269846766129683,
    0.26325757179864745,
    0.021677921540642985,
    0.0008991363869275422,
    0.0103893595005502,
    0.010369917040710183,
    0.04293744948226959,
    0.3181823601325353,
    0.6370094488064448,
    4.0602131706470255,
    0.06559762770775705,
    0.0655995136126876,
    0.00023938756977749114,
    1.007843542579311,
]
SCALER_STD = [
    7.224555550604913e-09,
    8.289663585957291e-09,
    5.572331607441521e-07,
    1.9699933543381123e-09,
    5.6897519634834455e-09,
    5.829530823285332e-09,
    9.749961107921822e-09,
    1.12463209860609e-08,
    6.093939052581492e-09,
    0.057515648382632324,
    0.2815839045442559,
    106.85689033389143,
    4.548367159964376,
    246.19255106699083,
    0.030232931812093616,
    0.03048513330391754,
    0.04129823224727286,
    0.015227951698918215,
    0.027472346034681464,
    0.0019105959352250694,
    0.0072249293101872124,
    0.007166316486238168,
    0.01949260887919908,
    1.292631986658928,
    0.14915231884175936,
    0.2307000978125421,
    0.02985657783421412,
    0.0298575991819556,
    0.00017911370253591535,
    0.16877502421633794,
]

PROB_THRESHOLD = 0.3110105052526263


def _load_iq_auto(filepath: str | Path) -> np.ndarray:
    data = np.load(filepath, mmap_mode="r")
    if np.iscomplexobj(data):
        n = min(MAX_IQ_SAMPLES, data.shape[0])
        return np.array(data[:n], dtype=np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        n = min(MAX_IQ_SAMPLES, data.shape[0])
        return (data[:n, 0].astype(np.float32) + 1j * data[:n, 1].astype(np.float32))
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        n = min(MAX_IQ_SAMPLES * 2, data.shape[0])
        return (data[:n:2].astype(np.float32) + 1j * data[1:n:2].astype(np.float32))
    raise ValueError(f"Unsupported format: shape={data.shape}, dtype={data.dtype}")


def _extract_features(iq: np.ndarray, sample_rate: float = 1e6) -> np.ndarray:
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

    # Vectorized spectral kurtosis via batched FFT
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


def evaluate(filename: str) -> int:
    iq = _load_iq_auto(filename)
    features = _extract_features(iq)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    w = np.array(LOGREG_WEIGHTS)
    mean = np.array(SCALER_MEAN)
    std = np.array(SCALER_STD)

    x_scaled = (features - mean) / (std + 1e-30)
    logit = float(np.dot(w, x_scaled) + LOGREG_BIAS)
    prob = 1.0 / (1.0 + np.exp(-logit))

    return 1 if prob >= PROB_THRESHOLD else 0

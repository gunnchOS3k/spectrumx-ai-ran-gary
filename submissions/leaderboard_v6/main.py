"""
v6 submission: PSD + LogReg with 30 features, 5M IQ, spectral kurtosis,
band energy ratios, and accuracy-tuned threshold.

Improvements over v5:
- 5M IQ samples (vs 2M) for better weak-signal detection
- 30 features (vs 21) including spectral kurtosis, band energies, autocorrelation
- Larger PSD window (8192) for finer frequency resolution
- Balanced CV accuracy: 91.9%
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

MAX_IQ_SAMPLES = 5_000_000

LOGREG_WEIGHTS = [
    0.08759717580169281,
    0.16303662119930304,
    0.2262827825774297,
    -0.07236536926984316,
    0.030816336413785627,
    0.027649247013768443,
    0.09526418984228342,
    0.1175288179363326,
    0.11689355099753805,
    -0.13547086836974767,
    -0.09013667674895542,
    0.1377627353734263,
    0.4134168563654617,
    0.3710728104004182,
    -0.8044180673832871,
    -0.6964783915690803,
    0.8322292928405475,
    0.7219198513663707,
    0.200043433932188,
    0.04933540540816722,
    0.08762387762203519,
    0.09104002472802136,
    0.05107740858368942,
    0.0428770759624755,
    0.039040734481433415,
    0.045508549680727255,
    0.05026967155227106,
    0.050555281940866864,
    -0.14404965235891456,
    0.04042327889696347,
]
LOGREG_BIAS = 1.6924336636225084

SCALER_MEAN = [
    1.0443095831147442e-08,
    3.912933025744062e-09,
    2.019647178127156e-07,
    3.0982425095379035e-09,
    8.820150038113538e-09,
    9.094809586921808e-09,
    1.2099512389692387e-08,
    1.3088520211164873e-08,
    3.004702811559845e-09,
    0.9739904214408134,
    12.940213187535603,
    21.13822594533904,
    0.7413873344659805,
    17.21666693687439,
    0.23361327891222536,
    0.23298738699475718,
    0.2700513103989592,
    0.26334802733137347,
    0.021930143157880635,
    0.0009364782156656777,
    0.010444205929040133,
    0.010400649091752711,
    0.042970737411330145,
    0.31229265928268435,
    0.6347242176532746,
    3.972698882735254,
    0.06570108788243184,
    0.06570230671204627,
    0.0003765775108722197,
    1.0054524870632728,
]
SCALER_STD = [
    7.503262896886031e-09,
    9.146807758296832e-09,
    5.89588858397217e-07,
    1.980619341283519e-09,
    5.6341736048453834e-09,
    5.8121683573203095e-09,
    1.0585471843554831e-08,
    1.2435072475530204e-08,
    7.252019580771705e-09,
    0.05952288291231517,
    0.28245287528768376,
    107.21109035375993,
    1.0954285045726173,
    51.26568405920801,
    0.030841719946668067,
    0.03088269773576794,
    0.04218750906181741,
    0.01552677884552128,
    0.028019070678711427,
    0.002284106287339826,
    0.00750298708902794,
    0.007324620641922573,
    0.01957330155036193,
    1.272816228271185,
    0.14654276171600494,
    0.22511036462425144,
    0.03008872183955889,
    0.030088575954683804,
    0.0002942437697040604,
    0.1642305468546909,
]

PROB_THRESHOLD = 0.29938938938938936


def _load_iq_auto(filepath: str | Path) -> np.ndarray:
    data = np.load(filepath, mmap_mode="r")
    if np.iscomplexobj(data):
        iq = data[:MAX_IQ_SAMPLES] if data.shape[0] > MAX_IQ_SAMPLES else data
        return np.array(iq, dtype=np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        n = min(MAX_IQ_SAMPLES, data.shape[0])
        return (data[:n, 0].astype(np.float32) + 1j * data[:n, 1].astype(np.float32))
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        n = min(MAX_IQ_SAMPLES * 2, data.shape[0])
        i = data[:n:2].astype(np.float32)
        q = data[1:n:2].astype(np.float32)
        return (i + 1j * q)
    raise ValueError(f"Unsupported format: shape={data.shape}, dtype={data.dtype}")


def _extract_features(iq: np.ndarray, sample_rate: float = 1e6) -> np.ndarray:
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

    n_segs = max(1, len(iq) // nperseg)
    if n_segs >= 4:
        seg_psds = []
        for i in range(n_segs):
            s = iq[i * nperseg: (i + 1) * nperseg]
            if len(s) < nperseg:
                break
            _, sp = sig.welch(s, fs=sample_rate, nperseg=min(1024, len(s)),
                              return_onesided=False, scaling="density")
            seg_psds.append(np.abs(sp))
        if len(seg_psds) >= 4:
            seg_arr = np.array(seg_psds)
            sk = scipy_kurtosis(seg_arr, axis=0, fisher=True)
            spectral_kurt_mean = float(np.mean(sk))
            spectral_kurt_max = float(np.max(sk))
        else:
            spectral_kurt_mean = 0.0
            spectral_kurt_max = 0.0
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

    ac = float(np.abs(np.mean(iq[1:] * np.conj(iq[:-1])))) if len(iq) > 1 else 0.0

    mag_kurtosis = float(scipy_kurtosis(mag, fisher=True))
    mag_skewness = float(scipy_skew(mag))
    crest_factor = float(np.max(mag) / (np.sqrt(mean_power) + 1e-30))
    i_std = float(np.std(iq.real))
    q_std = float(np.std(iq.imag))
    iq_corr = float(np.abs(np.corrcoef(iq.real, iq.imag)[0, 1])) if len(iq) > 1 else 0.0
    power_var_ratio = float(np.var(power)) / (mean_power ** 2 + 1e-30)

    return np.array([
        psd_mean, psd_std, psd_max, psd_min,
        float(np.percentile(psd_mag, 10)),
        float(np.percentile(psd_mag, 25)),
        float(np.percentile(psd_mag, 75)),
        float(np.percentile(psd_mag, 90)),
        float(np.percentile(psd_mag, 75) - np.percentile(psd_mag, 25)),
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

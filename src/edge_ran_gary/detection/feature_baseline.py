"""
Shared feature and IQ loading utilities for baseline detectors.

These functions are used both by:
    - local training scripts (e.g., train_feature_detector.py)
    - leaderboard submission wrapper (evaluate(filename))

They deliberately depend only on numpy/scipy so they can be reused in
lightweight environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import signal

DEFAULT_SAMPLE_RATE: float = 1e6  # Hz; aligned with 1-second windows at 1 MHz


def load_iq_from_npy(path: Path, is_int16_interleaved: bool = False) -> np.ndarray:
    """
    Load IQ data from a .npy file with support for multiple formats.

    Supported formats:
        - complex array, shape (N,), dtype complex64/complex128
        - float array, shape (N, 2), interpreted as [I, Q]
        - 1D float array, shape (N,)
        - int16 interleaved, shape (N*2,), interpreted as [I0, Q0, I1, Q1, ...]

    Returns:
        complex64 array of shape (N,)
    """
    data = np.load(str(path), allow_pickle=False)

    if is_int16_interleaved:
        if data.dtype != np.int16:
            raise ValueError(f"Expected int16 for interleaved format, got {data.dtype}")
        if len(data.shape) != 1 or data.shape[0] % 2 != 0:
            raise ValueError("Interleaved IQ data must be 1D with even length")
        i = data[::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        iq_complex = i + 1j * q
        return iq_complex.astype(np.complex64)

    # Already complex
    if np.iscomplexobj(data):
        if len(data.shape) != 1:
            raise ValueError(f"Complex data must be 1D, got shape {data.shape}")
        return data.astype(np.complex64)

    # Float array with columns [I, Q]
    if len(data.shape) == 2 and data.shape[1] == 2:
        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)
        iq_complex = i + 1j * q
        return iq_complex.astype(np.complex64)

    # 1D float: interpret as I-only, Q=0
    if len(data.shape) == 1 and np.issubdtype(data.dtype, np.floating):
        i = data.astype(np.float32)
        q = np.zeros_like(i)
        iq_complex = i + 1j * q
        return iq_complex.astype(np.complex64)

    raise ValueError(f"Unsupported data shape {data.shape}, dtype {data.dtype}")


def load_iq_auto(path: Path) -> np.ndarray:
    """
    Load IQ data from path, attempting standard formats.

    Strategy:
        1) Try complex/(N,2)/1D-float interpretation.
        2) If that fails, try int16 interleaved.
    """
    try:
        return load_iq_from_npy(path, is_int16_interleaved=False)
    except Exception:
        return load_iq_from_npy(path, is_int16_interleaved=True)


def compute_psd(iq: np.ndarray, sample_rate: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Compute two-sided PSD using Welch's method."""
    freqs, psd = signal.welch(
        iq,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density",
    )
    return freqs, psd


def extract_features(iq: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    Handcrafted feature extractor for 1-second IQ windows.

    Features:
        - mean_power
        - amp_variance
        - crest_factor
        - kurtosis
        - spectral_flatness
        - psd_entropy
        - topk_psd_peak_* (k=3)
        - band_energy_low/mid/high ratios
        - spectral_centroid
        - spectral_rolloff
        - acf_mean/var/max (short-lag autocorrelation on magnitude)
    """
    feats: Dict[str, float] = {}

    if iq.size == 0:
        return feats

    mag = np.abs(iq)
    power = mag ** 2

    # Time-domain statistics
    mean_power = float(np.mean(power))
    feats["mean_power"] = mean_power
    feats["amp_variance"] = float(np.var(mag))

    rms = float(np.sqrt(mean_power)) if mean_power > 0 else 0.0
    peak = float(np.max(mag))
    feats["crest_factor"] = float(peak / rms) if rms > 0 else 0.0

    # Kurtosis of magnitude
    m = float(np.mean(mag))
    if mag.size > 1:
        var = float(np.var(mag))
        if var > 0:
            centered = mag - m
            kurt = float(np.mean(centered ** 4) / (var ** 2))
        else:
            kurt = 0.0
    else:
        kurt = 0.0
    feats["kurtosis"] = kurt

    # Frequency-domain features
    freqs, psd = compute_psd(iq, sample_rate)
    psd_mag = np.abs(psd)
    total_power = float(np.sum(psd_mag))

    positive = psd_mag[psd_mag > 0]
    if positive.size > 0:
        geo_mean = float(np.exp(np.mean(np.log(positive))))
        arith_mean = float(np.mean(positive))
        sf = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0
    else:
        sf = 0.0
    feats["spectral_flatness"] = sf

    if total_power > 0 and psd_mag.size > 0:
        p = psd_mag / total_power
        psd_entropy = float(-np.sum(p * np.log(p + 1e-12)))
        feats["psd_entropy"] = psd_entropy

        # Top-k PSD peaks
        k = min(3, psd_mag.size)
        topk = np.sort(psd_mag)[-k:][::-1]
        for idx, val in enumerate(topk):
            feats[f"topk_psd_peak_{idx}"] = float(val)

        # Band energy ratios (low/mid/high split)
        abs_freqs = np.abs(freqs)
        low_band = abs_freqs < 0.2 * sample_rate
        mid_band = (abs_freqs >= 0.2 * sample_rate) & (abs_freqs < 0.4 * sample_rate)
        high_band = abs_freqs >= 0.4 * sample_rate
        e_low = float(np.sum(psd_mag[low_band]))
        e_mid = float(np.sum(psd_mag[mid_band]))
        e_high = float(np.sum(psd_mag[high_band]))
        denom = e_low + e_mid + e_high + 1e-12
        feats["band_energy_low_ratio"] = e_low / denom
        feats["band_energy_mid_ratio"] = e_mid / denom
        feats["band_energy_high_ratio"] = e_high / denom

        # Spectral centroid & rolloff
        centroid = float(np.sum(abs_freqs * psd_mag) / total_power)
        feats["spectral_centroid"] = centroid
        cumsum = np.cumsum(psd_mag)
        roll_frac = 0.95 * total_power
        idx = int(np.searchsorted(cumsum, roll_frac))
        if 0 <= idx < abs_freqs.size:
            feats["spectral_rolloff"] = float(abs_freqs[idx])
        else:
            feats["spectral_rolloff"] = float(abs_freqs[-1])
    else:
        feats["psd_entropy"] = 0.0
        feats["band_energy_low_ratio"] = 0.0
        feats["band_energy_mid_ratio"] = 0.0
        feats["band_energy_high_ratio"] = 0.0
        feats["spectral_centroid"] = 0.0
        feats["spectral_rolloff"] = 0.0
        for idx in range(3):
            feats[f"topk_psd_peak_{idx}"] = 0.0

    # Short-lag autocorrelation stats on magnitude
    max_lag = min(32, mag.size - 1) if mag.size > 1 else 0
    if max_lag > 0:
        acf_vals = []
        for lag in range(1, max_lag + 1):
            x1 = mag[:-lag]
            x2 = mag[lag:]
            if x1.size == 0:
                continue
            acf_vals.append(float(np.mean(x1 * x2)))
        if acf_vals:
            acf_arr = np.asarray(acf_vals, dtype=np.float64)
            feats["acf_mean"] = float(np.mean(acf_arr))
            feats["acf_var"] = float(np.var(acf_arr))
            feats["acf_max"] = float(np.max(acf_arr))
        else:
            feats["acf_mean"] = 0.0
            feats["acf_var"] = 0.0
            feats["acf_max"] = 0.0
    else:
        feats["acf_mean"] = 0.0
        feats["acf_var"] = 0.0
        feats["acf_max"] = 0.0

    return feats


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "load_iq_auto",
    "load_iq_from_npy",
    "compute_psd",
    "extract_features",
]


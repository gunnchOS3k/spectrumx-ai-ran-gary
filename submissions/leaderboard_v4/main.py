"""
Leaderboard v4 submission entrypoint.

Uses PSD + Logistic Regression with:
- 10 features (PSD stats + flatness + energy + peak ratio)
- Accuracy-tuned probability threshold (not default 0.5)
- Trained on full labeled dataset (240 files, 160 pos / 80 neg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

DEFAULT_SAMPLE_RATE = 1e6
MAX_IQ_SAMPLES = 1_000_000

# Trained weights (10 features) and bias
LOGREG_WEIGHTS = [
    2.993090778985938e-07,
    6.240837231629317e-08,
    7.977308885444927e-07,
    8.582491924994909e-08,
    2.5927600443959913e-07,
    3.549923985104825e-07,
    -0.40970235518648146,
    0.2995468607301177,
    0.8623068212421287,
    0.7990712687527355,
]
LOGREG_BIAS = -1.5132546994907476

# Accuracy-tuned threshold (sweept on labeled data, NOT default 0.5)
PROB_THRESHOLD = 0.2907035175879397


def _load_iq_from_npy(path: Path, is_int16_interleaved: bool = False) -> np.ndarray:
    data = np.load(str(path), allow_pickle=False)

    if is_int16_interleaved:
        if data.dtype != np.int16:
            raise ValueError(f"Expected int16, got {data.dtype}")
        if len(data.shape) != 1 or data.shape[0] % 2 != 0:
            raise ValueError("Interleaved IQ must be 1D with even length")
        i = data[::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)

    if np.iscomplexobj(data):
        if len(data.shape) != 1:
            raise ValueError(f"Complex data must be 1D, got shape {data.shape}")
        return data.astype(np.complex64)

    if len(data.shape) == 2 and data.shape[1] == 2:
        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)

    if len(data.shape) == 1 and np.issubdtype(data.dtype, np.floating):
        return (data.astype(np.float32) + 0j).astype(np.complex64)

    raise ValueError(f"Unsupported data shape {data.shape}, dtype {data.dtype}")


def _load_iq_auto(path: Path) -> np.ndarray:
    try:
        return _load_iq_from_npy(path, is_int16_interleaved=False)
    except Exception:
        return _load_iq_from_npy(path, is_int16_interleaved=True)


def _extract_features(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    _, psd = signal.welch(
        iq, fs=sample_rate, nperseg=1024, return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    if psd_mag.size == 0:
        return np.zeros(10, dtype=float)

    psd_pos = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    arith_mean = float(np.mean(psd_mag))
    flatness = geo_mean / arith_mean if arith_mean > 0 else 0.0

    mag = np.abs(iq)
    power = mag ** 2

    return np.array([
        np.mean(psd_mag),
        np.std(psd_mag),
        np.max(psd_mag),
        np.min(psd_mag),
        np.percentile(psd_mag, 25),
        np.percentile(psd_mag, 75),
        flatness,
        float(np.mean(power)),
        float(np.std(mag)),
        float(np.max(psd_mag) / (arith_mean + 1e-30)),
    ], dtype=float)


def evaluate(filename: str) -> int:
    try:
        path = Path(filename)
        if not path.is_file():
            return 0

        iq = _load_iq_auto(path)
        if iq is None or iq.size == 0 or iq.ndim != 1:
            return 0
        if MAX_IQ_SAMPLES and iq.size > MAX_IQ_SAMPLES:
            iq = iq[:MAX_IQ_SAMPLES]

        feats = _extract_features(iq, DEFAULT_SAMPLE_RATE)

        w = np.array(LOGREG_WEIGHTS, dtype=float)
        b = float(LOGREG_BIAS)
        logit = float(np.dot(w, feats) + b)
        prob = 1.0 / (1.0 + np.exp(-logit))

        return int(1 if prob >= PROB_THRESHOLD else 0)
    except Exception:
        return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/sample.npy")
        raise SystemExit(1)

    result = evaluate(sys.argv[1])
    print(int(result))

"""
Leaderboard baseline v2 submission entrypoint.

Defines `evaluate(filename)` as required by the competition:
    - loads a `.npy` IQ file robustly
    - runs tuned classical baselines
    - returns an integer 0 or 1

Design choices for v2:
    - Primary detector: spectral flatness with tuned threshold
    - Secondary detector: energy threshold with tuned threshold
    - Decision rule: predict occupied if either detector fires
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Tuned thresholds (from local sweep on labeled data)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 1e6  # Hz
ENERGY_THRESHOLD = 0.0012702238745987415
FLATNESS_THRESHOLD = 0.995281994342804

# ---------------------------------------------------------------------------
# IQ loader (adapted from apps/streamlit_app.py, without Streamlit dependency)
# ---------------------------------------------------------------------------


def _load_iq_from_npy(path: Path, is_int16_interleaved: bool = False) -> np.ndarray:
    """
    Load IQ data from a .npy file with support for multiple formats.
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

    if np.iscomplexobj(data):
        if len(data.shape) != 1:
            raise ValueError(f"Complex data must be 1D, got shape {data.shape}")
        return data.astype(np.complex64)

    if len(data.shape) == 2 and data.shape[1] == 2:
        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)
        iq_complex = i + 1j * q
        return iq_complex.astype(np.complex64)

    if len(data.shape) == 1 and np.issubdtype(data.dtype, np.floating):
        i = data.astype(np.float32)
        q = np.zeros_like(i)
        iq_complex = i + 1j * q
        return iq_complex.astype(np.complex64)

    raise ValueError(f"Unsupported data shape {data.shape}, dtype {data.dtype}")


def _load_iq_auto(path: Path) -> np.ndarray:
    """
    Load IQ data from path, attempting standard formats.
    """
    try:
        return _load_iq_from_npy(path, is_int16_interleaved=False)
    except Exception:
        return _load_iq_from_npy(path, is_int16_interleaved=True)


# ---------------------------------------------------------------------------
# Baseline detectors (mirroring src/edge_ran_gary/detection/baselines.py)
# ---------------------------------------------------------------------------


class EnergyDetector:
    """Simple energy detector baseline."""

    def __init__(self, threshold: float = ENERGY_THRESHOLD) -> None:
        self.threshold = float(threshold)

    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        power = float(np.mean(np.abs(iq) ** 2))
        prediction = 1 if power > self.threshold else 0
        distance = abs(power - self.threshold) / (self.threshold + 1e-10)
        confidence = float(min(1.0, distance))
        return int(prediction), confidence, power


class SpectralFlatnessDetector:
    """Spectral flatness detector baseline."""

    def __init__(self, threshold: float = FLATNESS_THRESHOLD, sample_rate: float = DEFAULT_SAMPLE_RATE) -> None:
        self.threshold = float(threshold)
        self.sample_rate = float(sample_rate)

    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        _, psd = signal.welch(
            iq,
            fs=self.sample_rate,
            nperseg=1024,
            return_onesided=False,
            scaling="density",
        )
        psd_mag = np.abs(psd)
        psd_mag = psd_mag[psd_mag > 0]

        if psd_mag.size == 0:
            return 0, 0.0, 0.0

        geometric_mean = float(np.exp(np.mean(np.log(psd_mag))))
        arithmetic_mean = float(np.mean(psd_mag))
        flatness = 0.0 if arithmetic_mean == 0.0 else geometric_mean / arithmetic_mean

        prediction = 1 if flatness < self.threshold else 0
        distance = abs(flatness - self.threshold) / (self.threshold + 1e-10)
        confidence = float(min(1.0, distance))
        return int(prediction), confidence, float(flatness)


# ---------------------------------------------------------------------------
# Public API: evaluate(filename) -> int
# ---------------------------------------------------------------------------


def evaluate(filename: str) -> int:
    """
    Competition entrypoint.
    """
    try:
        path = Path(filename)
        if not path.is_file():
            return 0

        iq = _load_iq_auto(path)
        if iq is None or iq.size == 0 or iq.ndim != 1:
            return 0

        sf_detector = SpectralFlatnessDetector()
        en_detector = EnergyDetector()

        pred_sf, _conf_sf, _flatness = sf_detector.detect(iq)
        pred_en, _conf_en, _power = en_detector.detect(iq)

        # OR rule: detect occupancy if either baseline fires
        return int(1 if (pred_sf == 1 or pred_en == 1) else 0)
    except Exception:
        return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/sample.npy")
        raise SystemExit(1)

    result = evaluate(sys.argv[1])
    print(int(result))

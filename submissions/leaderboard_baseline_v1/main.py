"""
Leaderboard baseline v1 submission entrypoint.

Defines `evaluate(filename)` as required by the competition:
    - loads a `.npy` IQ file robustly
    - runs a simple baseline detector
    - returns an integer 0 or 1

Design choices for v1:
    - Primary baseline: spectral flatness
    - Fallback baseline: energy detector

This file is intentionally self-contained so it can be zipped together
with `user_reqs.txt` without depending on the rest of the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# IQ loader (adapted from apps/streamlit_app.py, without Streamlit dependency)
# ---------------------------------------------------------------------------

def _load_iq_from_npy(path: Path, is_int16_interleaved: bool = False) -> np.ndarray:
    """
    Load IQ data from a .npy file with support for multiple formats.

    Supported formats:
        - complex array, shape (N,), dtype complex64/complex128
        - float array, shape (N, 2), interpreted as [I, Q]
        - 1D float array, shape (N,)
        - int16 interleaved, shape (N*2,), interpreted as [I0, Q0, I1, Q1, ...]

    Returns:
        complex64 array of shape (N,)

    Raises:
        ValueError on unsupported shapes/dtypes.
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


def _load_iq_auto(path: Path) -> np.ndarray:
    """
    Load IQ data from path, attempting standard formats.

    Strategy:
        1) Try complex/(N,2)/1D-float interpretation.
        2) If that fails, try int16 interleaved.
    """
    # First attempt: non-interleaved formats
    try:
        return _load_iq_from_npy(path, is_int16_interleaved=False)
    except Exception:
        # Second attempt: int16 interleaved
        return _load_iq_from_npy(path, is_int16_interleaved=True)


# ---------------------------------------------------------------------------
# Baseline detectors (mirroring src/edge_ran_gary/detection/baselines.py)
# ---------------------------------------------------------------------------

class EnergyDetector:
    """
    Simple energy detector baseline.

    Compares mean power to a threshold to determine occupancy.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = float(threshold)

    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        """
        Args:
            iq: complex64 array of shape (N,)

        Returns:
            (prediction, confidence, mean_power)
        """
        power = float(np.mean(np.abs(iq) ** 2))
        prediction = 1 if power > self.threshold else 0
        distance = abs(power - self.threshold) / (self.threshold + 1e-10)
        confidence = float(min(1.0, distance))
        return int(prediction), confidence, power


class SpectralFlatnessDetector:
    """
    Spectral flatness detector baseline.

    Uses spectral flatness (geometric mean / arithmetic mean of PSD)
    to distinguish structured signals from noise.
    Lower flatness indicates more signal-like behavior.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: float = 1e6) -> None:
        self.threshold = float(threshold)
        self.sample_rate = float(sample_rate)

    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        """
        Args:
            iq: complex64 array of shape (N,)

        Returns:
            (prediction, confidence, flatness)
        """
        freqs, psd = signal.welch(
            iq,
            fs=self.sample_rate,
            nperseg=1024,
            return_onesided=False,
            scaling="density",
        )
        psd_mag = np.abs(psd)
        psd_mag = psd_mag[psd_mag > 0]

        if psd_mag.size == 0:
            # Degenerate PSD, treat as noise / low confidence
            return 0, 0.0, 0.0

        geometric_mean = float(np.exp(np.mean(np.log(psd_mag))))
        arithmetic_mean = float(np.mean(psd_mag))

        if arithmetic_mean == 0.0:
            flatness = 0.0
        else:
            flatness = geometric_mean / arithmetic_mean

        prediction = 1 if flatness < self.threshold else 0
        distance = abs(flatness - self.threshold) / (self.threshold + 1e-10)
        confidence = float(min(1.0, distance))
        return int(prediction), confidence, float(flatness)


# ---------------------------------------------------------------------------
# Public API: evaluate(filename) -> int
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 1e6  # Hz; aligned with synthetic/micro-twin config


def evaluate(filename: str) -> int:
    """
    Competition entrypoint.

    Args:
        filename: Path to `.npy` IQ file.

    Returns:
        Integer 0 or 1:
            - 0: unoccupied / noise
            - 1: occupied / signal present
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Load IQ with robust format handling.
    iq = _load_iq_auto(path)
    if iq.ndim != 1:
        raise ValueError(f"Loaded IQ must be 1D, got shape {iq.shape}")

    # Primary baseline: spectral flatness.
    try:
        sf_detector = SpectralFlatnessDetector(
            threshold=0.5,
            sample_rate=DEFAULT_SAMPLE_RATE,
        )
        pred, _conf, _flatness = sf_detector.detect(iq)
        return int(1 if pred == 1 else 0)
    except Exception:
        # Fallback: energy detector.
        energy_detector = EnergyDetector(threshold=1.0)
        pred, _conf, _power = energy_detector.detect(iq)
        return int(1 if pred == 1 else 0)


if __name__ == "__main__":
    # Simple manual smoke test, not used by the competition harness.
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/sample.npy")
        raise SystemExit(1)

    result = evaluate(sys.argv[1])
    print(int(result))


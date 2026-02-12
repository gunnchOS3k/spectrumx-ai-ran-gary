"""
Data contract for Gary Micro-Twin v1 (competition-safe synthetic IQ only).

Stable schema for Ananya's ML pipeline: load iq -> predict -> prob_occupied.
"""

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


GENERATOR_VERSION = "1.0.0"

REQUIRED_METADATA_KEYS = (
    "sample_id",
    "zone_id",
    "landmark_name",
    "center_lat",
    "center_lon",
    "snr_db",
    "signal_type",
    "sample_rate_hz",
    "n_samples",
    "seed",
    "config_hash",
    "generator_version",
)


@dataclass
class DigitalTwinSample:
    """
    Single synthetic IQ sample with full metadata (competition-safe contract).

    - iq: complex64 (N,) — 1-second IQ window
    - label: 0 (noise only) or 1 (structured signal present)
    - metadata: dict with REQUIRED keys for reproducibility and ML consumption
    """
    iq: np.ndarray  # complex64, shape (N,)
    label: int      # 0 or 1
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.iq.dtype != np.complex64:
            self.iq = self.iq.astype(np.complex64)
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")
        missing = [k for k in REQUIRED_METADATA_KEYS if k not in self.metadata]
        if missing:
            raise ValueError(f"metadata missing required keys: {missing}")

    @property
    def n_samples(self) -> int:
        return int(self.metadata.get("n_samples", len(self.iq)))

    @property
    def sample_rate_hz(self) -> float:
        return float(self.metadata.get("sample_rate_hz", 0))

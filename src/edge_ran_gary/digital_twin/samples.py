"""
Generate DigitalTwinSample (stable contract) for Gary Micro-Twin v1.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .contracts import DigitalTwinSample, GENERATOR_VERSION, REQUIRED_METADATA_KEYS
from .generator import SignalGenerator, generate_iq_window
from .zones import ZoneModel


def _config_hash(config_path: str) -> str:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def generate_sample(
    seed: int,
    zone_id: str,
    label: Optional[int] = None,
    config_path: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> DigitalTwinSample:
    """
    Generate one competition-safe synthetic sample.

    Args:
        seed: Random seed (reproducibility).
        zone_id: Zone ID (must exist in config; invalid -> ValueError).
        label: 0 (noise), 1 (signal), or None (sample from zone occupancy_prior).
        config_path: Path to gary_micro_twin.yaml (required for zone lookup).
        sample_id: Optional sample ID; default f"{zone_id}_{seed}_{label}".

    Returns:
        DigitalTwinSample with iq, label, and full metadata.

    Raises:
        ValueError: If zone_id is not in config.
    """
    config_path = config_path or "configs/gary_micro_twin.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    sample_rate = float(config.get("sample_rate", 1e6))
    duration = float(config.get("duration", 1.0))
    n_samples = int(sample_rate * duration)
    config_hash = _config_hash(str(path))

    zone_model = ZoneModel.from_config(str(path))
    zone = zone_model.get_zone(zone_id)
    if zone is None:
        valid = ", ".join(zone_model.zone_ids)
        raise ValueError(f"Invalid zone_id '{zone_id}'. Valid zone IDs are: {valid}")

    rng = np.random.default_rng(seed)
    if label is None:
        label = int(rng.random() < zone.occupancy_prior)

    iq, raw_meta = generate_iq_window(
        seed=seed,
        label=label,
        config_path=str(path),
        sample_rate=sample_rate,
        duration=duration,
        zone_id=zone_id,
    )

    # Build contract metadata (all required keys)
    snr_db = raw_meta.get("snr_db")
    if snr_db is not None and not isinstance(snr_db, (int, float)):
        snr_db = float(snr_db) if snr_db is not None else None
    signal_type = raw_meta.get("signal_type", "noise") if label == 1 else "noise"

    metadata = {
        "sample_id": sample_id or f"{zone_id}_{seed}_{label}",
        "zone_id": zone.zone_id,
        "landmark_name": zone.landmark_name or zone.zone_id,
        "center_lat": zone.center_lat,
        "center_lon": zone.center_lon,
        "snr_db": snr_db,
        "signal_type": signal_type,
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "seed": seed,
        "config_hash": config_hash,
        "generator_version": GENERATOR_VERSION,
    }

    return DigitalTwinSample(iq=iq, label=label, metadata=metadata)

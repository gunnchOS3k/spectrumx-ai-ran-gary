"""Unit tests for Gary Micro-Twin: contract, invalid zone_id, SNR, QPSK complex."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = str(REPO_ROOT / "configs" / "gary_micro_twin.yaml")


def test_invalid_zone_id_raises():
    """Invalid zone_id must raise ValueError (no silent fallback)."""
    from edge_ran_gary.digital_twin.samples import generate_sample

    with pytest.raises(ValueError) as exc_info:
        generate_sample(seed=1, zone_id="invalid_zone_xyz", label=0, config_path=CONFIG)
    assert "Invalid zone_id" in str(exc_info.value)
    assert "invalid_zone_xyz" in str(exc_info.value)


def test_label1_snr_in_range():
    """Label=1 samples should have snr_db in metadata within ~2–3 dB of zone range."""
    from edge_ran_gary.digital_twin.samples import generate_sample

    s = generate_sample(seed=42, zone_id="city_hall", label=1, config_path=CONFIG)
    assert s.label == 1
    assert "snr_db" in s.metadata
    snr = s.metadata["snr_db"]
    assert snr is not None
    # city_hall snr_db_range [5, 20]
    assert 4 <= snr <= 21


def test_qpsk_has_nonzero_imaginary():
    """QPSK constellation must yield signal with non-zero imaginary components."""
    from edge_ran_gary.digital_twin.samples import generate_sample

    s = generate_sample(seed=99, zone_id="library", label=1, config_path=CONFIG)
    assert s.label == 1
    assert np.any(np.imag(s.iq) != 0), "QPSK/OFDM signal must have non-zero imaginary part"


def test_digital_twin_sample_contract():
    """DigitalTwinSample has required metadata keys."""
    from edge_ran_gary.digital_twin.contracts import DigitalTwinSample, REQUIRED_METADATA_KEYS
    from edge_ran_gary.digital_twin.samples import generate_sample

    s = generate_sample(seed=1, zone_id="high_school", label=0, config_path=CONFIG)
    assert isinstance(s, DigitalTwinSample)
    for k in REQUIRED_METADATA_KEYS:
        assert k in s.metadata, f"missing key {k}"
    assert s.metadata["zone_id"] == "high_school"
    assert s.metadata["generator_version"]

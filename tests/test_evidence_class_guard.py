"""Calibrated RF promotion must refuse without owner OTA provenance."""
from __future__ import annotations

from pathlib import Path

import pytest

from edge_ran_gary.evidence_class_guard import (
    EvidenceClassPromotionError,
    assert_evidence_class_allowed,
    ota_capture_bundle_present,
    stamp_report_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ota_bundle_absent():
    assert ota_capture_bundle_present(ROOT) is False


def test_synthetic_allowed():
    assert assert_evidence_class_allowed("SYNTHETIC_SIM", root=ROOT) == "SYNTHETIC_SIM"


def test_calibrated_refused():
    with pytest.raises(EvidenceClassPromotionError, match="refuse promoting"):
        assert_evidence_class_allowed("CALIBRATED", root=ROOT)


def test_fitted_calibrator_refused_without_bundle():
    with pytest.raises(EvidenceClassPromotionError, match="calibrator"):
        assert_evidence_class_allowed(
            "SYNTHETIC_SIM", root=ROOT, calibrator_fitted=True
        )


def test_submitted_refused():
    with pytest.raises(EvidenceClassPromotionError, match="SUBMITTED"):
        assert_evidence_class_allowed("SYNTHETIC_SIM", root=ROOT, submitted=True)


def test_stamp_keeps_synthetic():
    stamped = stamp_report_evidence(
        {"evidence_class": "SYNTHETIC_SIM", "calibration_applied": False},
        root=ROOT,
    )
    assert stamped["evidence_class"] == "SYNTHETIC_SIM"
    assert stamped["ota_captures_present"] is False
    assert stamped["calibrated_rf_present"] is False
    assert stamped["human_validation"] is False


def test_gap_ledger_honest():
    import json

    gap = json.loads((ROOT / "paper" / "CALIBRATED_RF_GAP.json").read_text())
    assert gap["evidence_class_current"] == "SYNTHETIC_SIM"
    assert gap["calibrated_rf_present"] is False
    assert gap["submitted_claim"] is False

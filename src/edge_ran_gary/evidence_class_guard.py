"""Refuse SYNTHETIC_SIM → CALIBRATED RF promotion without owner OTA/provenance.

Calibrator hooks may exist; fitting on invented physical labels is forbidden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

ALLOWED_WITHOUT_BUNDLE = frozenset({"SYNTHETIC_SIM", "UNCALIBRATED", "UNCALIBRATED_SYNTHETIC"})
CALIBRATED_CLASSES = frozenset({"CALIBRATED", "CALIBRATED_SIM", "FIELD_RF", "OTA_MEASURED"})


class EvidenceClassPromotionError(ValueError):
    """Raised when a report attempts calibrated RF class without provenance."""


def ota_capture_bundle_present(root: Path | None = None) -> bool:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    marker = base / "paper" / "ota_captures" / "PROVENANCE.sha256"
    if not marker.is_file():
        return False
    digest = marker.read_text(encoding="utf-8").strip()
    return len(digest) >= 64 and all(c in "0123456789abcdefABCDEF" for c in digest[:64])


def assert_evidence_class_allowed(
    evidence_class: str,
    *,
    root: Path | None = None,
    submitted: bool = False,
    calibrator_fitted: bool = False,
) -> str:
    cls = str(evidence_class or "").strip().upper()
    if submitted:
        raise EvidenceClassPromotionError(
            "SUBMITTED claim refused: packet forbids self-SUBMITTED without owner stamp"
        )
    if calibrator_fitted and not ota_capture_bundle_present(root):
        raise EvidenceClassPromotionError(
            "refuse calibrator.is_fitted=True without OTA/provenance capture bundle"
        )
    if cls in CALIBRATED_CLASSES and not ota_capture_bundle_present(root):
        raise EvidenceClassPromotionError(
            f"refuse promoting to {cls}: OTA capture bundle missing "
            "(paper/ota_captures/PROVENANCE.sha256)"
        )
    if cls not in ALLOWED_WITHOUT_BUNDLE and cls not in CALIBRATED_CLASSES:
        raise EvidenceClassPromotionError(f"unknown evidence_class={cls!r}")
    return cls


def stamp_report_evidence(
    report: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    out = dict(report)
    cls = assert_evidence_class_allowed(
        str(out.get("evidence_class", "SYNTHETIC_SIM")),
        root=root,
        submitted=bool(out.get("submitted") or out.get("submitted_claim")),
        calibrator_fitted=bool(out.get("calibration_applied") or out.get("calibrator_fitted")),
    )
    out["evidence_class"] = cls
    out["ota_captures_present"] = ota_capture_bundle_present(root)
    out["calibrated_rf_present"] = False if not out["ota_captures_present"] else bool(
        out.get("calibrated_rf_present")
    )
    out["human_validation"] = False
    return out

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
    """The ledger must name the commits it describes, not the moving HEAD.

    2e90a67 stored its parent (34941a5) in source_tip. Scientific tables are the
    646a8ae bytes. Results prose was corrected at 310c4b1. pdflatex is absent,
    so the PDF was not rebuilt. Original and revised vintages stay separate.
    """
    import json
    import shutil
    import subprocess

    gap = json.loads((ROOT / "paper" / "CALIBRATED_RF_GAP.json").read_text())
    tip = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True).strip()
    assert gap["evidence_class_current"] == "SYNTHETIC_SIM"
    assert gap["calibrated_rf_present"] is False
    assert gap["submitted_claim"] is False
    assert "HUMAN/RC2/V1/SUBMITTED remain false" in gap["non_claims"]

    def rev_parse(sha: str) -> str:
        return subprocess.check_output(["git", "rev-parse", sha], cwd=ROOT, text=True).strip()

    ledger = rev_parse(gap["source_tip"])
    tables = rev_parse(gap["scientific_tables_tip"])
    prose = rev_parse(gap["prose_correction_tip"])
    assert ledger == rev_parse("2e90a671fb0eb6ec8eaf68f5056cf6a60003e8d4")
    assert tables == rev_parse("646a8aeb34d34246eb7361abfd17f0f3b4526987")
    assert prose == rev_parse("310c4b17fe64f9ce227d3dacaf39124cfceb53af")
    assert gap["source_tip"] != tip
    historical = json.loads(
        subprocess.check_output(
            ["git", "show", f"{ledger}:paper/CALIBRATED_RF_GAP.json"],
            cwd=ROOT,
            text=True,
        )
    )
    assert historical["source_tip"].startswith("34941a5")
    assert gap["source_tip_recorded_in_that_commit"] == historical["source_tip"]

    for earlier, later in ((tables, prose), (ledger, prose), (prose, "HEAD")):
        subprocess.check_call(["git", "merge-base", "--is-ancestor", earlier, later], cwd=ROOT)

    table_delta = subprocess.check_output(
        ["git", "diff", "--name-only", tables, "HEAD", "--", "paper/tables"],
        cwd=ROOT,
        text=True,
    )
    assert table_delta.strip() == ""

    since_prose_parent = subprocess.check_output(
        ["git", "diff", "--name-only", f"{prose}^", "HEAD"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert "paper/sections/results.tex" in since_prose_parent
    assert not any(path.startswith("paper/tables/") for path in since_prose_parent)
    assert not any(path.endswith(".pdf") for path in since_prose_parent)
    vintage_roots = (
        "paper/submission/evidence/original/",
        "paper/submission/evidence/revised/",
        "paper/preprint/v0.1/evidence/data/original/",
        "paper/preprint/v0.1/evidence/data/revised/",
        "paper/preprint/v0.1/evidence/source/original/",
        "paper/preprint/v0.1/evidence/source/revised/",
    )
    assert not any(path.startswith(vintage_roots) for path in since_prose_parent)

    prose_paths = subprocess.check_output(
        ["git", "diff", "--name-only", f"{prose}^", prose],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert "paper/sections/results.tex" in prose_paths
    assert not any(path.startswith("paper/tables/") or path.endswith(".pdf") for path in prose_paths)

    def blob(rev: str, path: str) -> str:
        return subprocess.check_output(["git", "rev-parse", f"{rev}:{path}"], cwd=ROOT, text=True).strip()

    for pdf in ("paper/submission/manuscript.pdf", "paper/preprint/v0.1/preprint.pdf"):
        assert blob(f"{prose}^", pdf) == blob("HEAD", pdf)

    assert gap["pdf_rebuilt"] is False
    assert gap["pdflatex_present"] is False
    assert gap["result_vintages_mixed"] is False
    assert shutil.which("pdflatex") is None
    assert any("pdflatex is absent" in line for line in gap["non_claims"])

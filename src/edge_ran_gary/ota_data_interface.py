"""
Aerial Data Lake / OTA evidence **interface** (completed extension, **OTA-ready target**).

Defines **what** over-the-air captures would contain for **retraining, calibration,
and evaluation** without claiming captures exist in-repo.

**Judged detector:** still trained/evaluated on official SpectrumX IQ offline unless
you explicitly add OTA-backed datasets to your pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OTACaptureRecord:
    """Logical fields for one OTA IQ / metadata record (schema target)."""

    capture_id: str
    site_id: str
    center_freq_hz: float
    sample_rate_hz: float
    duration_s: float
    antenna_config: str
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    weather_tag: Optional[str] = None
    interference_tags: List[str] = field(default_factory=list)
    detector_label_at_capture: Optional[int] = None
    ground_truth_label: Optional[int] = None
    calibration_offset_db: Optional[float] = None
    notes: Optional[str] = None


def default_ota_schema_dict() -> Dict[str, Any]:
    """JSON-serializable overview for docs and Streamlit."""
    ex = OTACaptureRecord(
        capture_id="ota_gary_0001",
        site_id="public_library",
        center_freq_hz=3.5e9,
        sample_rate_hz=1e6,
        duration_s=1.0,
        antenna_config="single_port_omni",
        detector_label_at_capture=0,
        ground_truth_label=0,
    )
    return {
        "schema_version": 1,
        "purpose": (
            "Map OTA RF captures to **detector retraining**, **threshold calibration**, "
            "and **digital-twin replay** (Gary anchors)."
        ),
        "example_record": asdict(ex),
        "recommended_files": [
            "data/ota_evidence/ota_lake_manifest.json",
            "data/ota_evidence/captures/*.iqmeta.json + companion .npy",
        ],
    }


def load_ota_lake_manifest(repo_root: Path) -> Dict[str, Any]:
    """
    Load ``data/ota_evidence/ota_lake_manifest.json`` if present and minimally valid.

    Returns fields consumed by ``finalize_simulation_status`` in hooks.
    """
    root = repo_root.resolve()
    d = root / "data" / "ota_evidence"
    p = d / "ota_lake_manifest.json"
    primary = str(d.resolve())

    if not p.is_file():
        return {
            "loaded": False,
            "path": primary,
            "integration": "ota_data_lake",
            "source_kind": "absent",
            "status_label": "OTA target (not active)",
            "expected_files": ["ota_lake_manifest.json"],
            "error": None,
        }

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {
            "loaded": False,
            "path": str(p.resolve()),
            "integration": "ota_data_lake",
            "source_kind": "absent",
            "status_label": "Not loaded",
            "error": str(e),
        }

    if not isinstance(raw, dict) or not raw.get("ota_lake_manifest_version"):
        return {
            "loaded": False,
            "path": str(p.resolve()),
            "integration": "ota_data_lake",
            "source_kind": "absent",
            "status_label": "Not loaded",
            "error": "validation_failed",
        }

    return {
        "loaded": True,
        "path": str(p.resolve()),
        "data": raw,
        "integration": "ota_data_lake",
        "source_kind": "ota",
        "status_label": "OTA-backed evidence (manifest)",
        "ota_backed": True,
        "error": None,
    }


def map_ota_to_detector_training_notes() -> List[str]:
    return [
        "Use **ground_truth_label** vs **detector_label_at_capture** for calibration drift tables.",
        "Join **site_id** + **weather_tag** with **Gary scenario engine** presets for domain shift analysis.",
        "Feed **IQ .npy** paths listed in manifest into offline retraining (not Streamlit Cloud).",
    ]

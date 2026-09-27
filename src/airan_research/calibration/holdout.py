"""Synthetic calibration + holdout. Never overwrites parent model versions."""
from __future__ import annotations

from typing import Any


def calibrate(
    optimization: dict[str, Any],
    measurement: dict[str, float],
    *,
    parent_model_version: str = "CAMPUS_V2_PLANNING",
    producer_commit: str = "0" * 40,
) -> dict[str, Any]:
    pred = {
        "coverage": optimization["objectives"]["coverage"],
        "capacity": optimization["objectives"]["capacity"],
        "latency": optimization["objectives"]["latency"],
        "jitter": optimization["objectives"]["jitter"],
        "packet_loss": optimization["objectives"]["packet_loss"],
    }
    residual = {k: round(measurement[k] - pred[k], 4) for k in pred}
    holdout_error = round(sum(abs(v) for v in residual.values()) / len(residual), 4)
    new_version = f"CALIBRATED_SIM_{optimization['site_id']}_V1"
    if new_version == parent_model_version:
        raise RuntimeError("refusing to overwrite parent model version")
    commit = producer_commit if len(producer_commit) == 40 else "0" * 40
    return {
        "schema_name": "gunnchos.twin_calibration_bundle",
        "schema_version": "1.0.0",
        "run_id": f"planning-cal-{optimization['site_id']}",
        "site_id": optimization["site_id"],
        "campus_slug": optimization["campus_slug"],
        "phase": optimization["phase"],
        "source_manifest_sha256": optimization["source_manifest_sha256"],
        "parent_model_version": parent_model_version,
        "new_model_version": new_version,
        "prediction": pred,
        "measurement": measurement,
        "residual": residual,
        "uncertainty": {
            "overall": "high",
            "notes": "Synthetic holdout only. Not real twin calibration.",
        },
        "candidate_parameter_update": {
            "parameters": {
                "path_loss_exponent": 2.15,
                "attenuation_db": 5.5,
                "blockage_db": 7.5,
                "effective_node_range_m": 18.0,
                "backhaul_latency_ms": 12.0,
                "aggregate_demand_factor": 0.95,
                "failover_timing_s": 8.0,
            },
            "applied": True,
            "notes": "New version only; parent retained.",
        },
        "holdout_validation": {
            "holdout_fraction": 0.25,
            "holdout_error": holdout_error,
            "pass": holdout_error < 5.0,
            "n_train": 3,
            "n_holdout": 1,
        },
        "rejected_evidence": [
            {
                "evidence_ref": "quarantine://invalid",
                "reason": "quarantined or identifier-bearing evidence is rejected",
            }
        ],
        "evidence_class": "SIMULATED",
        "producer": {"repository": "spectrumx-ai-ran-gary", "commit": commit},
    }

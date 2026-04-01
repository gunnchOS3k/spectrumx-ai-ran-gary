"""
Explicit **evidence / provenance** and **execution surface** for the research extension.

**Judged SpectrumX detector** is unchanged; these labels apply only to simulation / OTA / bridge hooks.

Canonical **evidence** vocabulary (user-facing, six terms)
---------------------------------------------------------
Use ``canonical_evidence_status`` / ``provenance_label`` on hook results:

- ``proxy-only`` — Twin/scenario uses analytic proxies; no validated solver export (or DM/SN stub path).
- ``loaded demo`` — Valid manifest from ``examples/`` or downgraded ``export_provenance`` (not ``full_solver``).
- ``loaded simulation export`` — Valid ``data/``-style export treated as operator or full_solver-backed.
- ``installer-ready`` — Aerial access probe only (``access_summary.json``); not a twin export.
- ``external-runtime-required`` — Integration target (pyAerial, AODT generation, OTA capture) needs GPU/lab/cloud; no honest in-app execution.
- ``OTA-backed`` — OTA lake manifest present and valid.

**Execution surface** (orthogonal): where code runs — ``manifest-load-only`` | ``external-runtime-required`` | ``lab-OTA-workflow``.

Machine keys (``provenance_tier``) stay stable for JSON/logs; user-facing strings use the vocabulary above.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- Machine tier keys (stable; backward compatible) ---
PROXY_ONLY = "proxy_only"
DEMO_SUMMARY_LOADED = "demo_summary_loaded"
SIMULATION_EXPORT_LOADED = "simulation_export_loaded"
ACCESS_INSTALLER_READY = "access_installer_ready"
INTEGRATION_READY_STUB = "integration_ready_stub"
OTA_BACKED = "ota_backed"

# --- Canonical evidence strings (exact display vocabulary) ---
CANON_PROXY_ONLY = "proxy-only"
CANON_LOADED_DEMO = "loaded demo"
CANON_LOADED_SIMULATION_EXPORT = "loaded simulation export"
CANON_INSTALLER_READY = "installer-ready"
CANON_EXTERNAL_RUNTIME_REQUIRED = "external-runtime-required"
CANON_OTA_BACKED = "OTA-backed"

# --- Canonical execution surface strings ---
CANON_EXEC_MANIFEST_ONLY = "manifest-load-only"
CANON_EXEC_EXTERNAL_RUNTIME = "external-runtime-required"
CANON_EXEC_LAB_OTA = "lab-OTA-workflow"

# Execution surface (machine keys; internal)
EXECUTION_STREAMLIT_MANIFEST = "streamlit_manifest_load_only"
EXECUTION_EXTERNAL_RUNTIME = "external_runtime_required"
EXECUTION_LAB_OTA = "lab_ota_workflow"

# Long-form help for docs / Streamlit expanders (keyed by canonical evidence term)
CANONICAL_EVIDENCE_HELP: Dict[str, str] = {
    CANON_PROXY_ONLY: "Scenario engine and maps use **proxies**; no validated DeepMIMO/Sionna/AODT export on disk for this layer.",
    CANON_LOADED_DEMO: "Parsed JSON/GeoJSON from **bundled examples** or analytic export scripts (not ``full_solver`` provenance).",
    CANON_LOADED_SIMULATION_EXPORT: "Parsed artifact under ``data/`` with **simulation** lineage (operator file or ``full_solver`` provenance).",
    CANON_INSTALLER_READY: "**access_summary.json** from env probe — credentials/installer posture only; **not** a scene solve.",
    CANON_EXTERNAL_RUNTIME_REQUIRED: "Credible **PHY / twin / OTA IQ** for this layer requires **external** GPU, lab, or cloud; Streamlit does not execute it.",
    CANON_OTA_BACKED: "**ota_lake_manifest.json** loaded; IQ binaries and labeling remain **outside** Streamlit.",
}

CANONICAL_EXECUTION_HELP: Dict[str, str] = {
    CANON_EXEC_MANIFEST_ONLY: "**This app:** loads and displays manifests/summaries only.",
    CANON_EXEC_EXTERNAL_RUNTIME: "**Regeneration / full solve:** requires external MATLAB, Sionna GPU, Omniverse, Aerial, or batch host.",
    CANON_EXEC_LAB_OTA: "**Captures:** lab or field hardware; manifests may be read in-app.",
}

# Legacy: internal tier -> short description for maintainers (not always equal to canonical evidence)
INTERNAL_TIER_DESCRIPTIONS: Dict[str, str] = {
    PROXY_ONLY: "Internal: proxy_only",
    DEMO_SUMMARY_LOADED: "Internal: demo_summary_loaded",
    SIMULATION_EXPORT_LOADED: "Internal: simulation_export_loaded",
    ACCESS_INSTALLER_READY: "Internal: access_installer_ready",
    INTEGRATION_READY_STUB: "Internal: integration_ready_stub",
    OTA_BACKED: "Internal: ota_backed",
}

# Backward compatibility: old name used by imports
TIER_LABELS: Dict[str, str] = dict(INTERNAL_TIER_DESCRIPTIONS)

EXECUTION_SURFACE_LABELS: Dict[str, str] = {
    EXECUTION_STREAMLIT_MANIFEST: CANONICAL_EXECUTION_HELP[CANON_EXEC_MANIFEST_ONLY],
    EXECUTION_EXTERNAL_RUNTIME: CANONICAL_EXECUTION_HELP[CANON_EXEC_EXTERNAL_RUNTIME],
    EXECUTION_LAB_OTA: CANONICAL_EXECUTION_HELP[CANON_EXEC_LAB_OTA],
}


def provenance_tier_from_sim_result(result: Dict[str, Any]) -> str:
    """Map hook result to internal machine tier key."""
    if not isinstance(result, dict):
        return PROXY_ONLY
    if result.get("ota_backed") or (
        str(result.get("integration")) == "ota_data_lake" and result.get("loaded")
    ):
        return OTA_BACKED
    if str(result.get("integration")) == "pyaerial_bridge":
        if not result.get("loaded"):
            return INTEGRATION_READY_STUB
        sk = str(result.get("source_kind") or "absent")
        if sk == "demo":
            return DEMO_SUMMARY_LOADED
        if sk == "simulation":
            return SIMULATION_EXPORT_LOADED
        return INTEGRATION_READY_STUB
    sk = str(result.get("source_kind") or "absent")
    loaded = bool(result.get("loaded"))
    if sk == "access" or result.get("access_confirmed"):
        return ACCESS_INSTALLER_READY
    if not loaded:
        return INTEGRATION_READY_STUB
    if sk == "demo":
        return DEMO_SUMMARY_LOADED
    if sk == "simulation":
        return SIMULATION_EXPORT_LOADED
    return INTEGRATION_READY_STUB


def evidence_tier_to_canonical(tier: str, integration: str) -> str:
    """Map internal tier + integration id to one of the six canonical evidence terms."""
    integ = str(integration or "")
    if tier == OTA_BACKED:
        return CANON_OTA_BACKED
    if tier == ACCESS_INSTALLER_READY:
        return CANON_INSTALLER_READY
    if tier == DEMO_SUMMARY_LOADED:
        return CANON_LOADED_DEMO
    if tier == SIMULATION_EXPORT_LOADED:
        return CANON_LOADED_SIMULATION_EXPORT
    if tier == PROXY_ONLY:
        return CANON_PROXY_ONLY
    if tier == INTEGRATION_READY_STUB:
        if integ in ("pyaerial_bridge", "aerial_omniverse", "ota_data_lake"):
            return CANON_EXTERNAL_RUNTIME_REQUIRED
        return CANON_PROXY_ONLY
    return CANON_EXTERNAL_RUNTIME_REQUIRED


def execution_tier_to_canonical(ex_tier: str) -> str:
    if ex_tier == EXECUTION_LAB_OTA:
        return CANON_EXEC_LAB_OTA
    if ex_tier == EXECUTION_EXTERNAL_RUNTIME:
        return CANON_EXEC_EXTERNAL_RUNTIME
    return CANON_EXEC_MANIFEST_ONLY


def provenance_label_for_tier(tier: str, integration: str = "") -> str:
    """User-facing canonical evidence label (prefer over raw internal tier in UI)."""
    return evidence_tier_to_canonical(tier, integration)


def attach_provenance_tier(result: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``provenance_tier``, ``provenance_label``, and ``canonical_evidence_status``."""
    tier = provenance_tier_from_sim_result(result)
    result = dict(result)
    integ = str(result.get("integration") or "")
    result["provenance_tier"] = tier
    canon = evidence_tier_to_canonical(tier, integ)
    result["provenance_label"] = canon
    result["canonical_evidence_status"] = canon
    return result


def attach_execution_surface(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Declares Streamlit vs external runtime. Sets ``execution_surface_*`` and
    ``canonical_execution_status``.
    """
    r = dict(result)
    integ = str(r.get("integration") or "")
    loaded = bool(r.get("loaded"))

    streamlit_line = CANONICAL_EXECUTION_HELP[CANON_EXEC_MANIFEST_ONLY]

    if integ == "ota_data_lake":
        ex_tier = EXECUTION_LAB_OTA
        if loaded:
            ext_line = (
                "**OTA:** manifest loaded for **provenance**; IQ and labeling are **lab/field**; "
                "retraining/evaluation are **offline**."
            )
        else:
            ext_line = (
                "**OTA:** no active manifest — use ``examples/ota_evidence/ota_lake_manifest.example.json`` as a template."
            )
    elif integ == "deepmimo":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**DeepMIMO:** regenerate with ``scripts/export_deepmimo_summary.py`` or an external pipeline; "
            "``scripts/validate_simulation_exports.py --deepmimo`` checks shape."
        )
    elif integ == "sionna_rt":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**Sionna RT:** analytic script or **GPU** ray trace off-box; "
            "``scripts/validate_simulation_exports.py --sionna`` checks shape."
        )
    elif integ == "aerial_omniverse":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**AODT:** full scene in Omniverse / AI Aerial + **GPU**; repo reads **manifests** only "
            "(``docs/AODT_EXPORT_CHECKLIST.md``)."
        )
    elif integ == "pyaerial_bridge":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**pyAerial / cuPHY:** NVIDIA Aerial stack on **GPU/container**; app shows **manifest + import probe** only."
        )
    else:
        ex_tier = EXECUTION_STREAMLIT_MANIFEST
        ext_line = "No integration id on this record."

    r["execution_surface_tier"] = ex_tier
    r["canonical_execution_status"] = execution_tier_to_canonical(ex_tier)

    if ex_tier == EXECUTION_STREAMLIT_MANIFEST:
        r["execution_surface_label"] = f"{streamlit_line} {ext_line}"
    elif ex_tier == EXECUTION_LAB_OTA:
        r["execution_surface_label"] = (
            f"{streamlit_line} {CANONICAL_EXECUTION_HELP[CANON_EXEC_LAB_OTA]} {ext_line}"
        )
    else:
        r["execution_surface_label"] = (
            f"{streamlit_line} {CANONICAL_EXECUTION_HELP[CANON_EXEC_EXTERNAL_RUNTIME]} {ext_line}"
        )
    return r


def finalize_simulation_status(result: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance + execution surface (use from hooks)."""
    return attach_execution_surface(attach_provenance_tier(result))


def civic_stack_summary(
    deepmimo: Dict[str, Any],
    sionna: Dict[str, Any],
    aerial: Dict[str, Any],
    pyaerial: Dict[str, Any],
    ota: Dict[str, Any],
) -> Dict[str, Any]:
    """One JSON-serializable row for expanders / architecture cards."""
    return {
        "deepmimo": {
            "provenance_tier": provenance_tier_from_sim_result(deepmimo),
            "label": deepmimo.get("canonical_evidence_status")
            or deepmimo.get("provenance_label")
            or evidence_tier_to_canonical(
                provenance_tier_from_sim_result(deepmimo), str(deepmimo.get("integration") or "deepmimo")
            ),
        },
        "sionna_rt": {
            "provenance_tier": provenance_tier_from_sim_result(sionna),
            "label": sionna.get("canonical_evidence_status")
            or sionna.get("provenance_label")
            or evidence_tier_to_canonical(
                provenance_tier_from_sim_result(sionna), str(sionna.get("integration") or "sionna_rt")
            ),
        },
        "aerial_aodt": {
            "provenance_tier": provenance_tier_from_sim_result(aerial),
            "label": aerial.get("canonical_evidence_status")
            or aerial.get("provenance_label")
            or evidence_tier_to_canonical(
                provenance_tier_from_sim_result(aerial), str(aerial.get("integration") or "aerial_omniverse")
            ),
            "aerial_status_tier": aerial.get("aerial_status_tier"),
        },
        "pyaerial_bridge": {
            "provenance_tier": pyaerial.get("provenance_tier", INTEGRATION_READY_STUB),
            "label": pyaerial.get("canonical_evidence_status")
            or pyaerial.get("provenance_label")
            or evidence_tier_to_canonical(
                provenance_tier_from_sim_result(pyaerial), str(pyaerial.get("integration") or "pyaerial_bridge")
            ),
        },
        "ota_data_lake": {
            "provenance_tier": ota.get("provenance_tier", INTEGRATION_READY_STUB),
            "label": ota.get("canonical_evidence_status")
            or ota.get("provenance_label")
            or evidence_tier_to_canonical(
                provenance_tier_from_sim_result(ota), str(ota.get("integration") or "ota_data_lake")
            ),
        },
    }


def ordered_canonical_evidence_terms() -> List[str]:
    """Fixed order for legends and docs."""
    return [
        CANON_PROXY_ONLY,
        CANON_LOADED_DEMO,
        CANON_LOADED_SIMULATION_EXPORT,
        CANON_INSTALLER_READY,
        CANON_EXTERNAL_RUNTIME_REQUIRED,
        CANON_OTA_BACKED,
    ]

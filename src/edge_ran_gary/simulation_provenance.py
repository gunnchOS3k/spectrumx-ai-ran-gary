"""
Explicit **realism / evidence provenance** labels for the completed research extension.

These tiers are **UI and documentation vocabulary** for Streamlit and roadmap docs.
They do **not** change judged SpectrumX scoring.

Use with ``simulation_integration_hooks`` load results: each integration returns
``status_label``, ``source_kind``, and optional ``provenance_tier``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Canonical tier keys (machine)
PROXY_ONLY = "proxy_only"
DEMO_SUMMARY_LOADED = "demo_summary_loaded"
SIMULATION_EXPORT_LOADED = "simulation_export_loaded"
ACCESS_INSTALLER_READY = "access_installer_ready"
INTEGRATION_READY_STUB = "integration_ready_stub"
OTA_BACKED = "ota_backed"

# Human labels (Streamlit / docs) — no em dash in titles per style guide
TIER_LABELS: Dict[str, str] = {
    PROXY_ONLY: "Proxy only (scenario math, not solver-backed)",
    DEMO_SUMMARY_LOADED: "Loaded (demo summary)",
    SIMULATION_EXPORT_LOADED: "Loaded (simulation export)",
    ACCESS_INSTALLER_READY: "Access confirmed / installer-ready",
    INTEGRATION_READY_STUB: "Integration-ready (stub / manifest only)",
    OTA_BACKED: "OTA-backed evidence (capture manifest present)",
}


def provenance_tier_from_sim_result(result: Dict[str, Any]) -> str:
    """
    Map a hook result dict (DeepMIMO / Sionna / Aerial) to a canonical tier key.
    """
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


def provenance_label_for_tier(tier: str) -> str:
    return TIER_LABELS.get(tier, tier)


def attach_provenance_tier(result: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates and returns ``result`` with ``provenance_tier`` and ``provenance_label``."""
    tier = provenance_tier_from_sim_result(result)
    result = dict(result)
    result["provenance_tier"] = tier
    result["provenance_label"] = provenance_label_for_tier(tier)
    return result


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
            "label": deepmimo.get("status_label", provenance_label_for_tier(provenance_tier_from_sim_result(deepmimo))),
        },
        "sionna_rt": {
            "provenance_tier": provenance_tier_from_sim_result(sionna),
            "label": sionna.get("status_label", provenance_label_for_tier(provenance_tier_from_sim_result(sionna))),
        },
        "aerial_aodt": {
            "provenance_tier": provenance_tier_from_sim_result(aerial),
            "label": aerial.get("status_label", provenance_label_for_tier(provenance_tier_from_sim_result(aerial))),
            "aerial_status_tier": aerial.get("aerial_status_tier"),
        },
        "pyaerial_bridge": {
            "provenance_tier": pyaerial.get("provenance_tier", INTEGRATION_READY_STUB),
            "label": pyaerial.get("status_label", "Not loaded"),
        },
        "ota_data_lake": {
            "provenance_tier": ota.get("provenance_tier", INTEGRATION_READY_STUB),
            "label": ota.get("status_label", "OTA target (not active)"),
        },
    }

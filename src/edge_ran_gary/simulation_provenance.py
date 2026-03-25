"""
Explicit **realism / evidence provenance** labels for the completed research extension.

These tiers are **UI and documentation vocabulary** for Streamlit and roadmap docs.
They do **not** change judged SpectrumX scoring.

Use with ``simulation_integration_hooks`` load results: each integration returns
``status_label``, ``source_kind``, ``provenance_tier``, and ``execution_surface_*``
(what Streamlit loads vs what requires **external** GPU / lab / OTA capture).
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

# Execution surface (orthogonal to file provenance): what Streamlit does vs external infrastructure.
EXECUTION_STREAMLIT_MANIFEST = "streamlit_manifest_load_only"
EXECUTION_EXTERNAL_RUNTIME = "external_runtime_required"
EXECUTION_LAB_OTA = "lab_ota_workflow"

# Human labels (Streamlit / docs) — no em dash in titles per style guide
TIER_LABELS: Dict[str, str] = {
    PROXY_ONLY: "Proxy only (scenario math, not solver-backed)",
    DEMO_SUMMARY_LOADED: "Loaded (demo summary)",
    SIMULATION_EXPORT_LOADED: "Loaded (simulation export)",
    ACCESS_INSTALLER_READY: "Access confirmed / installer-ready",
    INTEGRATION_READY_STUB: "Integration-ready (stub / manifest only)",
    OTA_BACKED: "OTA-backed evidence (capture manifest present)",
}

EXECUTION_SURFACE_LABELS: Dict[str, str] = {
    EXECUTION_STREAMLIT_MANIFEST: (
        "Streamlit Cloud / this app: **visualization and manifest load only** — no solvers, PHY, or OTA capture execution."
    ),
    EXECUTION_EXTERNAL_RUNTIME: (
        "Full generation or refresh: **external** GPU, MATLAB, Omniverse, Aerial, or batch host "
        "(see `docs/EXTERNAL_RUNTIME_GAPS.md`)."
    ),
    EXECUTION_LAB_OTA: (
        "OTA IQ and labels: **lab or field** workflow; this app may read **manifest JSON** only."
    ),
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


def attach_execution_surface(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Declares what runs in Streamlit vs what **must** run externally (GPU / lab / installer).

    Orthogonal to ``provenance_tier`` (demo vs simulation export vs OTA-backed, etc.).
    """
    r = dict(result)
    integ = str(r.get("integration") or "")
    loaded = bool(r.get("loaded"))

    streamlit_line = EXECUTION_SURFACE_LABELS[EXECUTION_STREAMLIT_MANIFEST]

    if integ == "ota_data_lake":
        ex_tier = EXECUTION_LAB_OTA
        if loaded:
            ext_line = (
                "**OTA:** manifest is loaded for **provenance**; IQ binaries and labeling live **outside** Streamlit. "
                "Retraining and evaluation are **offline** batch jobs."
            )
        else:
            ext_line = (
                "**OTA:** **not active**. Use `examples/ota_evidence/ota_lake_manifest.example.json` as a template; "
                "captures are **lab/field** only."
            )
    elif integ == "deepmimo":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**DeepMIMO:** regenerate `scenario_summary.json` with `scripts/export_deepmimo_summary.py` or an external "
            "DeepMIMO/MATLAB pipeline; validate with `scripts/validate_simulation_exports.py --deepmimo`."
        )
    elif integ == "sionna_rt":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**Sionna RT:** regenerate summaries with `scripts/export_sionna_rt_summary.py` (analytic) or a **GPU** "
            "Sionna RT ray-tracing job; validate with `scripts/validate_simulation_exports.py --sionna`."
        )
    elif integ == "aerial_omniverse":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**AODT:** **full-scene generation** requires NVIDIA Omniverse / AI Aerial tooling and **GPU**; "
            "this repo consumes **export manifests** only (`docs/AODT_EXPORT_CHECKLIST.md`)."
        )
    elif integ == "pyaerial_bridge":
        ex_tier = EXECUTION_EXTERNAL_RUNTIME
        ext_line = (
            "**pyAerial / cuPHY:** runtime is **NVIDIA Aerial** stack (container or bare-metal **GPU**). "
            "Streamlit shows **bridge manifest + import probe** only (`docs/PYAERIAL_BRIDGE.md`)."
        )
    else:
        ex_tier = EXECUTION_STREAMLIT_MANIFEST
        ext_line = "**No heavy integration id** on this record."

    r["execution_surface_tier"] = ex_tier
    if ex_tier == EXECUTION_STREAMLIT_MANIFEST:
        r["execution_surface_label"] = f"{streamlit_line} {ext_line}"
    elif ex_tier == EXECUTION_LAB_OTA:
        r["execution_surface_label"] = f"{streamlit_line} {EXECUTION_SURFACE_LABELS[ex_tier]} {ext_line}"
    else:
        r["execution_surface_label"] = (
            f"{streamlit_line} {EXECUTION_SURFACE_LABELS[EXECUTION_EXTERNAL_RUNTIME]} {ext_line}"
        )
    return r


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

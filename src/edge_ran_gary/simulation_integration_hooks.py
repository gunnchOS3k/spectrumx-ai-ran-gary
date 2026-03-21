"""
Integration hooks for simulation-backed realism (completed extension only).

This module does **not** run DeepMIMO, Sionna RT, or NVIDIA AI Aerial / Omniverse.
It documents **drop zones** and optional future loaders for the Streamlit extension.

Truthfulness:
- Judged SpectrumX detector is separate; nothing here drives official scoring.
- All loaders return ``None`` until real parsers exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Primary drop zones (repo root–relative) ---
DATA_DEEPMIMO_REL = Path("data/deepmimo")
DATA_SIONNA_RT_REL = Path("data/sionna_rt")
DATA_AERIAL_OMNIVERSE_REL = Path("data/aerial_omniverse")
CONFIG_WIRELESS_SCENE_REL = Path("configs/wireless_scene")
CONFIG_RIC_REL = Path("configs/ric")

# --- Legacy / alternate paths (still monitored) ---
LEGACY_DEEPMIMO_REL = Path("data/simulation/deepmimo")
LEGACY_SIONNA_RT_REL = Path("data/simulation/sionna_rt")

DEEPMIMO_SUGGESTED_FILES = (
    "scenario_meta.json",
    "channel_features.npz",
    "site_sinr_proxy.csv",
)

SIONNA_SUGGESTED_FILES = (
    "materials.yaml",
    "coverage_grid.geojson",
    "path_loss_summary.json",
)

AERIAL_SUGGESTED_FILES = (
    "scene_export.usd",
    "twin_manifest.json",
    "rf_overlay_meta.yaml",
)

WIRELESS_SCENE_CONFIG_SUGGESTED = (
    "gary_anchor_sites.yaml",
    "layer_visibility.yaml",
)

RIC_CONFIG_SUGGESTED = (
    "xapp_policy_stub.yaml",
    "near_rt_ric_endpoints.example.yaml",
)


def _dir_status(d: Path) -> Dict[str, Any]:
    exists = d.is_dir()
    files: List[str] = []
    if exists:
        try:
            files = sorted(p.name for p in d.iterdir() if p.is_file())[:40]
        except OSError:
            files = []
    return {"exists": exists, "path": str(d.resolve()), "files": files}


def deepmimo_drop_dirs(repo_root: Path) -> List[Path]:
    r = repo_root.resolve()
    return [r / DATA_DEEPMIMO_REL, r / LEGACY_DEEPMIMO_REL]


def sionna_rt_drop_dirs(repo_root: Path) -> List[Path]:
    r = repo_root.resolve()
    return [r / DATA_SIONNA_RT_REL, r / LEGACY_SIONNA_RT_REL]


def aerial_drop_dir(repo_root: Path) -> Path:
    return (repo_root / DATA_AERIAL_OMNIVERSE_REL).resolve()


def wireless_scene_config_dir(repo_root: Path) -> Path:
    return (repo_root / CONFIG_WIRELESS_SCENE_REL).resolve()


def ric_config_dir(repo_root: Path) -> Path:
    return (repo_root / CONFIG_RIC_REL).resolve()


def _merge_dir_status(paths: List[Path], suggested: tuple[str, ...]) -> Dict[str, Any]:
    """First existing dir wins for 'primary'; list all paths."""
    statuses = [_dir_status(p) for p in paths]
    primary = next((s for s in statuses if s["exists"]), statuses[0] if statuses else {})
    artifact_like: List[str] = []
    for s in statuses:
        if not s.get("exists"):
            continue
        for f in s.get("files", []):
            if f.endswith((".npz", ".npy", ".json", ".csv", ".geojson", ".yaml", ".yml", ".usd")):
                artifact_like.append(f)
    return {
        "paths_checked": [str(p.resolve()) for p in paths],
        "primary": primary,
        "suggested_inputs": list(suggested),
        "artifact_like": sorted(set(artifact_like))[:25],
    }


def describe_simulation_backbone_status(repo_root: Path) -> Dict[str, Any]:
    """Backward-compatible summary + extended keys for UI/JSON expander."""
    return describe_all_integration_drop_zones(repo_root)


def describe_all_integration_drop_zones(repo_root: Path) -> Dict[str, Any]:
    """Full integration map: data drops + config stubs (honest exists/files)."""
    r = repo_root.resolve()
    dm = _merge_dir_status(deepmimo_drop_dirs(r), DEEPMIMO_SUGGESTED_FILES)
    sn = _merge_dir_status(sionna_rt_drop_dirs(r), SIONNA_SUGGESTED_FILES)
    ae = _dir_status(aerial_drop_dir(r))
    ae_hits = [f for f in ae["files"] if f.endswith((".json", ".yaml", ".yml", ".usd", ".usda"))]
    ws = _dir_status(wireless_scene_config_dir(r))
    ric = _dir_status(ric_config_dir(r))
    return {
        "deepmimo": {
            **dm,
            "status": "integration_ready_next_scaling",
            "note": "Channel / CSI artifacts for site-specific scenarios — not active in UI yet.",
        },
        "sionna_rt": {
            **sn,
            "status": "integration_ready_next_scaling",
            "note": "Ray-traced propagation / coverage GeoJSON — not active in UI yet.",
        },
        "nvidia_ai_aerial_omniverse": {
            **ae,
            "suggested_inputs": list(AERIAL_SUGGESTED_FILES),
            "artifact_like": ae_hits,
            "status": "integration_ready_next_scaling",
            "note": "Digital-twin / RF-visualization exports — requires separate Aerial/Omniverse tooling; not bundled.",
        },
        "configs_wireless_scene": {
            **ws,
            "suggested_inputs": list(WIRELESS_SCENE_CONFIG_SUGGESTED),
            "status": "optional_config",
        },
        "configs_ric": {
            **ric,
            "suggested_inputs": list(RIC_CONFIG_SUGGESTED),
            "status": "optional_config",
        },
    }


def describe_extension_asset_status(repo_root: Path) -> Dict[str, Any]:
    """Optional local files + integration drop zones (flat paths for UI lists)."""
    root = repo_root.resolve()
    return {
        "submission_metrics_csv": {
            "path": str(root / "submissions" / "submission_metrics.csv"),
            "exists": (root / "submissions" / "submission_metrics.csv").is_file(),
        },
        "final_report_figures_yaml": {
            "path": str(root / "docs" / "final_report_figures.yaml"),
            "exists": (root / "docs" / "final_report_figures.yaml").is_file(),
        },
        "gary_micro_twin_yaml": {
            "path": str(root / "configs" / "gary_micro_twin.yaml"),
            "exists": (root / "configs" / "gary_micro_twin.yaml").is_file(),
        },
        "data_deepmimo": {
            "path": str((root / DATA_DEEPMIMO_REL).resolve()),
            "exists": (root / DATA_DEEPMIMO_REL).is_dir(),
        },
        "data_sionna_rt": {
            "path": str((root / DATA_SIONNA_RT_REL).resolve()),
            "exists": (root / DATA_SIONNA_RT_REL).is_dir(),
        },
        "data_aerial_omniverse": {
            "path": str(aerial_drop_dir(root)),
            "exists": aerial_drop_dir(root).is_dir(),
        },
        "configs_wireless_scene": {
            "path": str(wireless_scene_config_dir(root)),
            "exists": wireless_scene_config_dir(root).is_dir(),
        },
        "configs_ric": {
            "path": str(ric_config_dir(root)),
            "exists": ric_config_dir(root).is_dir(),
        },
        "legacy_data_simulation_deepmimo": {
            "path": str((root / LEGACY_DEEPMIMO_REL).resolve()),
            "exists": (root / LEGACY_DEEPMIMO_REL).is_dir(),
        },
        "legacy_data_simulation_sionna": {
            "path": str((root / LEGACY_SIONNA_RT_REL).resolve()),
            "exists": (root / LEGACY_SIONNA_RT_REL).is_dir(),
        },
    }


def load_deepmimo_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Future: load from data/deepmimo or data/simulation/deepmimo."""
    for d in deepmimo_drop_dirs(repo_root):
        if d.is_dir():
            return None
    return None


def load_sionna_rt_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Future: load GeoJSON / JSON from data/sionna_rt or legacy path."""
    for d in sionna_rt_drop_dirs(repo_root):
        if d.is_dir():
            return None
    return None


def load_aerial_omniverse_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Future: load Aerial / Omniverse digital-twin RF overlay metadata."""
    d = aerial_drop_dir(repo_root)
    if d.is_dir():
        return None
    return None


# Back-compat names used by older docs
def deepmimo_dir(repo_root: Path) -> Path:
    p = repo_root / LEGACY_DEEPMIMO_REL
    if p.is_dir():
        return p.resolve()
    return (repo_root / DATA_DEEPMIMO_REL).resolve()


def sionna_rt_dir(repo_root: Path) -> Path:
    p = repo_root / LEGACY_SIONNA_RT_REL
    if p.is_dir():
        return p.resolve()
    return (repo_root / DATA_SIONNA_RT_REL).resolve()

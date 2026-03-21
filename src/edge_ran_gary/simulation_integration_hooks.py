"""
Placeholders and status helpers for next-stage wireless simulation integration.

This module does **not** run DeepMIMO or Sionna RT. It documents expected local paths
and optional future file drops so the Streamlit extension can load real outputs when available.

Truthfulness:
- Judged SpectrumX detector is separate; simulation outputs here are for the **completed extension** only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Expected drop directories (create locally; optional .gitkeep in repo)
DEEPMIMO_REL = Path("data/simulation/deepmimo")
SIONNA_RT_REL = Path("data/simulation/sionna_rt")

# Suggested filenames (convention — adjust when pipelines exist)
DEEPMIMO_SUGGESTED_FILES = (
    "scenario_meta.json",  # site IDs, carrier, bandwidth, seed
    "channel_features.npz",  # or .npy — CIR summaries / features for ML
    "site_sinr_proxy.csv",  # optional tabular overlay for UI
)

SIONNA_SUGGESTED_FILES = (
    "materials.yaml",  # radio materials / permittivity hooks
    "coverage_grid.geojson",  # optional map overlay
    "path_loss_summary.json",  # per-link or per-tile summaries
)


def deepmimo_dir(repo_root: Path) -> Path:
    return (repo_root / DEEPMIMO_REL).resolve()


def sionna_rt_dir(repo_root: Path) -> Path:
    return (repo_root / SIONNA_RT_REL).resolve()


def _dir_status(d: Path) -> Dict[str, Any]:
    exists = d.is_dir()
    files: List[str] = []
    if exists:
        try:
            files = sorted(p.name for p in d.iterdir() if p.is_file())[:30]
        except OSError:
            files = []
    return {"exists": exists, "path": str(d), "files": files}


def describe_simulation_backbone_status(repo_root: Path) -> Dict[str, Any]:
    """Return honest status for UI: dirs exist, any matching suggested artifacts."""
    dm = _dir_status(deepmimo_dir(repo_root))
    sn = _dir_status(sionna_rt_dir(repo_root))
    dm_hits = [f for f in dm["files"] if f.endswith((".npz", ".npy", ".json", ".csv"))]
    sn_hits = [f for f in sn["files"] if f.endswith((".json", ".geojson", ".yaml", ".yml"))]
    return {
        "deepmimo": {**dm, "suggested_inputs": list(DEEPMIMO_SUGGESTED_FILES), "artifact_like": dm_hits},
        "sionna_rt": {**sn, "suggested_inputs": list(SIONNA_SUGGESTED_FILES), "artifact_like": sn_hits},
    }


def describe_extension_asset_status(repo_root: Path) -> Dict[str, Any]:
    """Optional local files the industry-grade extension can reference (no competition IQ)."""
    root = repo_root.resolve()
    paths = {
        "submission_metrics_csv": root / "submissions" / "submission_metrics.csv",
        "final_report_figures_yaml": root / "docs" / "final_report_figures.yaml",
        "gary_micro_twin_yaml": root / "configs" / "gary_micro_twin.yaml",
        "deepmimo_dir": deepmimo_dir(root),
        "sionna_rt_dir": sionna_rt_dir(root),
    }
    out: Dict[str, Any] = {}
    for key, p in paths.items():
        out[key] = {"path": str(p), "exists": p.is_file() if key != "deepmimo_dir" and key != "sionna_rt_dir" else p.is_dir()}
    # dirs
    out["deepmimo_dir"]["exists"] = paths["deepmimo_dir"].is_dir()
    out["sionna_rt_dir"]["exists"] = paths["sionna_rt_dir"].is_dir()
    return out


def load_deepmimo_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    """
    Future: load NPZ/JSON from data/simulation/deepmimo for map overlays or feature panels.
    Returns None until real loader is implemented.
    """
    d = deepmimo_dir(repo_root)
    if not d.is_dir():
        return None
    # Placeholder: no parser yet
    return None


def load_sionna_rt_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    """
    Future: load GeoJSON or JSON summaries from data/simulation/sionna_rt for coverage layers.
    Returns None until real loader is implemented.
    """
    d = sionna_rt_dir(repo_root)
    if not d.is_dir():
        return None
    return None

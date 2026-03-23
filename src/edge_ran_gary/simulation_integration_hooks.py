"""
Integration hooks for simulation-backed realism (completed extension only).

This module does **not** run DeepMIMO, Sionna RT, or NVIDIA AI Aerial / Omniverse.
It **loads and parses** optional on-disk summaries and artifacts when present.

Truthfulness:
- ``loaded: True`` only when a **recognized** JSON/GeoJSON parse succeeds — never faked.
- Judged SpectrumX detector is separate; nothing here drives official scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Summary JSON filenames (first match wins per directory scan order)
DEEPMIMO_SUMMARY_FILES = ("scenario_summary.json", "scenario_meta.json")
SIONNA_SUMMARY_FILES = ("propagation_summary.json", "path_loss_summary.json")
AERIAL_SUMMARY_FILES = ("overlay_summary.json", "twin_manifest.json")

SIONNA_GEOJSON_NAMES = ("coverage_grid.geojson", "coverage.geojson", "sionna_coverage.geojson")


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
            "note": "Channel / CSI artifacts for site-specific scenarios — optional JSON/NPZ parsers in hooks.",
        },
        "sionna_rt": {
            **sn,
            "status": "integration_ready_next_scaling",
            "note": "Ray-traced propagation / coverage GeoJSON — optional parsers in hooks.",
        },
        "nvidia_ai_aerial_omniverse": {
            **ae,
            "suggested_inputs": list(AERIAL_SUGGESTED_FILES),
            "artifact_like": ae_hits,
            "status": "integration_ready_next_scaling",
            "note": "Requires **external** NVIDIA AI Aerial / Omniverse tooling + often **NVIDIA account** / **6G Developer Program** access; not bundled.",
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


# ---------------------------------------------------------------------------
# DeepMIMO — parse + optional NPZ metadata
# ---------------------------------------------------------------------------


def _scan_npz_metadata(npz_path: Path) -> Optional[Dict[str, Any]]:
    if not npz_path.is_file():
        return None
    try:
        import numpy as np

        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
        keys = list(z.files)[:20]
        shapes = {}
        for k in keys[:12]:
            try:
                arr = z[k]
                shapes[k] = list(arr.shape)
            except Exception:
                shapes[k] = "?"
        z.close()
        return {"path": str(npz_path.resolve()), "arrays": shapes}
    except Exception as e:
        return {"path": str(npz_path.resolve()), "error": str(e)}


def _normalize_deepmimo_dict(data: Any) -> Dict[str, Any]:
    """Flatten common DeepMIMO export shapes into a display row."""
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    root = data.get("deepmimo") if isinstance(data.get("deepmimo"), dict) else data
    if not isinstance(root, dict):
        root = data

    out["scenario_name"] = (
        root.get("scenario_name")
        or root.get("name")
        or root.get("scenario")
        or root.get("dataset_name")
    )
    out["num_bs"] = root.get("num_bs") or root.get("num_base_stations") or root.get("M") or root.get("n_bs")
    out["num_users"] = root.get("num_users") or root.get("num_ue") or root.get("K") or root.get("n_ue")
    out["num_antennas_bs"] = root.get("num_antennas_bs") or root.get("n_tx")
    out["num_antennas_ue"] = root.get("num_antennas_ue") or root.get("n_rx")
    out["los_links"] = root.get("los_links") or root.get("n_los")
    out["nlos_links"] = root.get("nlos_links") or root.get("n_nlos")
    out["carrier_ghz"] = root.get("carrier_ghz") or root.get("fc_ghz")
    out["version"] = root.get("version") or data.get("version")
    out["generated_at"] = root.get("generated_at") or root.get("export_time")

    stats = root.get("channel_statistics") or root.get("statistics")
    if isinstance(stats, dict):
        for k in ("mean_path_loss_db", "median_sinr_db", "mean_sinr_db"):
            if k in stats:
                out[k] = stats[k]

    sites = root.get("sites") or data.get("sites")
    if isinstance(sites, list):
        out["num_sites"] = len(sites)
    elif isinstance(sites, dict):
        out["num_sites"] = len(sites)

    return {k: v for k, v in out.items() if v is not None}


def _deepmimo_parse_valid(extracted: Dict[str, Any], raw: Dict[str, Any]) -> bool:
    """True only if we have enough signal that this is a real summary, not an empty {}."""
    if extracted.get("scenario_name"):
        return True
    if extracted.get("num_bs") is not None or extracted.get("num_users") is not None:
        return True
    if extracted.get("los_links") is not None or extracted.get("nlos_links") is not None:
        return True
    if raw.get("deepmimo_export_version") or raw.get("deepmimo_version"):
        return True
    if isinstance(raw.get("channel_statistics"), dict) and len(raw["channel_statistics"]) > 0:
        return True
    if isinstance(raw.get("sites"), (list, dict)) and len(raw["sites"]) > 0:
        return True
    return False


def load_deepmimo_scenario_summary(repo_root: Path) -> Dict[str, Any]:
    """
    Load and **validate** DeepMIMO JSON summary + optional ``channel_features.npz`` metadata.
    ``loaded`` is True only if JSON parses and passes field validation.
    """
    r = repo_root.resolve()
    dirs = deepmimo_drop_dirs(r)
    primary_path = ""
    if dirs:
        primary_path = str(dirs[0].resolve())

    raw_data: Optional[Dict[str, Any]] = None
    json_path: Optional[Path] = None
    last_err: Optional[str] = None

    for d in dirs:
        if not d.is_dir():
            continue
        for fn in DEEPMIMO_SUMMARY_FILES:
            p = d / fn
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                raw_data = obj if isinstance(obj, dict) else {"value": obj}
                json_path = p
                break
            except json.JSONDecodeError as e:
                last_err = f"JSON decode ({p.name}): {e}"
            except OSError as e:
                last_err = str(e)
        if json_path:
            break

    npz_meta = None
    if json_path is not None:
        parent = json_path.parent
        for cand in ("channel_features.npz", "channels.npz", "deepmimo_channels.npz"):
            m = _scan_npz_metadata(parent / cand)
            if m and "arrays" in m:
                npz_meta = m
                break

    if raw_data is None:
        return {
            "loaded": False,
            "path": primary_path,
            "expected_files": list(DEEPMIMO_SUMMARY_FILES),
            "expected_schema": (
                "JSON object with e.g. scenario_name, num_bs, num_users, los_links/nlos_links, "
                "or channel_statistics / sites — see docs/SIMULATION_BACKBONE_PLAN.md"
            ),
            "error": last_err,
            "integration": "deepmimo",
            "parser": "deepmimo_v1",
        }

    extracted = _normalize_deepmimo_dict(raw_data)
    ok = _deepmimo_parse_valid(extracted, raw_data)
    return {
        "loaded": ok,
        "path": str(json_path.resolve()),
        "data": raw_data,
        "extracted_summary": extracted,
        "npz_channel_meta": npz_meta,
        "integration": "deepmimo",
        "parser": "deepmimo_v1",
        "parser_note": None
        if ok
        else "File is valid JSON but missing expected DeepMIMO summary fields — not marked loaded.",
        "error": None if ok else (last_err or "validation_failed"),
    }


# ---------------------------------------------------------------------------
# Sionna RT — JSON summary + optional GeoJSON coverage grid
# ---------------------------------------------------------------------------


def _find_first_file(dirs: List[Path], names: tuple[str, ...]) -> Optional[Path]:
    for d in dirs:
        if not d.is_dir():
            continue
        for fn in names:
            p = d / fn
            if p.is_file():
                return p
    return None


def _geojson_quick_stats(path: Path) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, {}, str(e)
    t = g.get("type")
    feats = g.get("features")
    if t == "FeatureCollection" and isinstance(feats, list):
        return True, {"type": t, "feature_count": len(feats)}, None
    if t == "Feature" and g.get("geometry"):
        return True, {"type": t, "feature_count": 1}, None
    return False, {}, "Not a FeatureCollection/Feature GeoJSON"


def _normalize_sionna_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    root = data.get("sionna") if isinstance(data.get("sionna"), dict) else data
    if not isinstance(root, dict):
        root = data

    out["scenario_name"] = root.get("scenario_name") or root.get("name") or root.get("scenario")
    out["frequency_ghz"] = root.get("frequency_ghz") or root.get("fc_ghz")
    out["mean_path_loss_db"] = root.get("mean_path_loss_db")
    out["median_path_loss_db"] = root.get("median_path_loss_db")
    out["p10_path_loss_db"] = root.get("p10_path_loss_db")
    out["p90_path_loss_db"] = root.get("p90_path_loss_db")
    out["los_fraction"] = root.get("los_fraction") or root.get("los_ratio")
    out["materials_summary"] = root.get("materials_summary") or root.get("materials")
    out["blockage_model"] = root.get("blockage_model")
    out["version"] = root.get("version") or data.get("version")
    out["generated_at"] = root.get("generated_at")

    cov = root.get("coverage") or root.get("coverage_stats")
    if isinstance(cov, dict):
        for k in ("mean_sinr_db", "cell_edge_sinr_db", "fraction_above_threshold"):
            if k in cov:
                out[f"coverage_{k}"] = cov[k]

    return {k: v for k, v in out.items() if v is not None}


def _sionna_parse_valid(extracted: Dict[str, Any], raw: Dict[str, Any]) -> bool:
    if extracted.get("scenario_name"):
        return True
    if any(
        extracted.get(k) is not None
        for k in ("mean_path_loss_db", "median_path_loss_db", "los_fraction", "frequency_ghz")
    ):
        return True
    if raw.get("sionna_export_version") or raw.get("solver"):
        return True
    if isinstance(raw.get("cells"), list) and len(raw["cells"]) > 0:
        return True
    return False


def load_sionna_propagation_summary(repo_root: Path) -> Dict[str, Any]:
    """
    Load Sionna RT **propagation_summary** / **path_loss_summary** JSON and/or **coverage_grid.geojson**.
    ``loaded`` is True if JSON validates **or** GeoJSON validates (coverage-only export).
    """
    r = repo_root.resolve()
    dirs = sionna_rt_drop_dirs(r)
    primary_path = str(dirs[0].resolve()) if dirs else ""

    raw_data: Optional[Dict[str, Any]] = None
    json_path: Optional[Path] = None
    last_err: Optional[str] = None

    for d in dirs:
        if not d.is_dir():
            continue
        for fn in SIONNA_SUMMARY_FILES:
            p = d / fn
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                raw_data = obj if isinstance(obj, dict) else {"value": obj}
                json_path = p
                break
            except json.JSONDecodeError as e:
                last_err = f"JSON decode ({p.name}): {e}"
            except OSError as e:
                last_err = str(e)
        if json_path:
            break

    gj_path = _find_first_file(dirs, SIONNA_GEOJSON_NAMES)
    geo_ok = False
    geo_stats: Dict[str, Any] = {}
    geo_err: Optional[str] = None
    geojson_inline: Optional[Dict[str, Any]] = None
    if gj_path is not None:
        geo_ok, geo_stats, geo_err = _geojson_quick_stats(gj_path)
        if geo_ok:
            try:
                geojson_inline = json.loads(gj_path.read_text(encoding="utf-8"))
            except Exception:
                geojson_inline = None

    extracted: Dict[str, Any] = {}
    json_ok = False
    if raw_data is not None:
        extracted = _normalize_sionna_dict(raw_data)
        json_ok = _sionna_parse_valid(extracted, raw_data)

    loaded = json_ok or geo_ok
    combined_extracted = {**extracted}
    if geo_ok:
        combined_extracted["coverage_geojson_features"] = geo_stats.get("feature_count")
        combined_extracted["coverage_geojson_type"] = geo_stats.get("type")

    parser_note = None
    if not loaded:
        parser_note = "No valid Sionna summary JSON and no valid coverage GeoJSON found."
    elif not json_ok and geo_ok:
        parser_note = "Loaded from **GeoJSON only** — add propagation_summary.json for richer metrics."

    return {
        "loaded": loaded,
        "path": str(json_path.resolve()) if json_path else (str(gj_path.resolve()) if gj_path else primary_path),
        "summary_json_path": str(json_path.resolve()) if json_path else None,
        "geojson_path": str(gj_path.resolve()) if gj_path and geo_ok else None,
        "data": raw_data if raw_data is not None else {},
        "extracted_summary": combined_extracted,
        "geojson_stats": geo_stats if geo_ok else {},
        "geojson_for_deck": geojson_inline,
        "integration": "sionna_rt",
        "parser": "sionna_rt_v1",
        "parser_note": parser_note,
        "error": None if loaded else (last_err or geo_err or "not_found"),
        "expected_files": list(SIONNA_SUMMARY_FILES) + list(SIONNA_GEOJSON_NAMES),
        "expected_schema": (
            "JSON: scenario_name / path loss / LOS fraction / coverage block; "
            "or GeoJSON FeatureCollection for coverage_grid — see docs/SIMULATION_BACKBONE_PLAN.md"
        ),
    }


# ---------------------------------------------------------------------------
# Aerial / Omniverse — JSON summaries only; no fake content
# ---------------------------------------------------------------------------


def _normalize_aerial_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["scene_name"] = data.get("scene_name") or data.get("name") or data.get("twin_name")
    out["usd_path"] = data.get("usd_path") or data.get("scene_export")
    out["version"] = data.get("version")
    out["generated_at"] = data.get("generated_at")
    out["notes"] = data.get("notes")
    return {k: v for k, v in out.items() if v is not None}


def _aerial_parse_valid(extracted: Dict[str, Any], raw: Dict[str, Any]) -> bool:
    if extracted.get("scene_name") or extracted.get("usd_path"):
        return True
    if raw.get("aerial_export_version") or raw.get("omniverse_kit_version"):
        return True
    if isinstance(raw.get("assets"), list) and len(raw["assets"]) > 0:
        return True
    return False


def load_aerial_overlay_summary(repo_root: Path) -> Dict[str, Any]:
    """Load twin_manifest / overlay_summary JSON only if it passes validation."""
    d = aerial_drop_dir(repo_root)
    raw_data: Optional[Dict[str, Any]] = None
    json_path: Optional[Path] = None
    last_err: Optional[str] = None

    for fn in AERIAL_SUMMARY_FILES:
        p = d / fn
        if not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            raw_data = obj if isinstance(obj, dict) else {"value": obj}
            json_path = p
            break
        except json.JSONDecodeError as e:
            last_err = f"JSON decode ({fn}): {e}"
        except OSError as e:
            last_err = str(e)

    if raw_data is None:
        return {
            "loaded": False,
            "path": str(d.resolve()),
            "expected_files": list(AERIAL_SUMMARY_FILES),
            "expected_schema": (
                "JSON with scene_name / usd_path or aerial_export_version — "
                "**NVIDIA AI Aerial / Omniverse** runs **outside** this repo; "
                "full fidelity needs **external program access**, **GPU**, and often **NVIDIA account** / **6G Developer Program**."
            ),
            "integration": "aerial_omniverse",
            "parser": "aerial_v1",
            "external_tooling_required": True,
            "error": last_err,
        }

    extracted = _normalize_aerial_dict(raw_data)
    ok = _aerial_parse_valid(extracted, raw_data)
    return {
        "loaded": ok,
        "path": str(json_path.resolve()) if json_path else str(d.resolve()),
        "data": raw_data,
        "extracted_summary": extracted,
        "integration": "aerial_omniverse",
        "parser": "aerial_v1",
        "external_tooling_required": True,
        "parser_note": None
        if ok
        else "JSON present but missing expected manifest fields — not marked loaded.",
        "error": None if ok else "validation_failed",
    }


def load_deepmimo_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    s = load_deepmimo_scenario_summary(repo_root)
    return s if s.get("loaded") else None


def load_sionna_rt_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    s = load_sionna_propagation_summary(repo_root)
    return s if s.get("loaded") else None


def load_aerial_omniverse_overlay_stub(repo_root: Path) -> Optional[Dict[str, Any]]:
    s = load_aerial_overlay_summary(repo_root)
    return s if s.get("loaded") else None


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

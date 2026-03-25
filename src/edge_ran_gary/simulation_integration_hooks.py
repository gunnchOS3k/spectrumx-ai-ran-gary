"""
Integration hooks for simulation-backed realism (completed extension only).

This module does **not** run full DeepMIMO, Sionna RT, or NVIDIA AI Aerial / Omniverse solvers.
It **loads and parses** optional on-disk summaries and artifacts when present.

See ``scripts/export_*`` and ``docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`` for generating ``data/*`` exports;
``export_provenance.simulation_grade`` controls **demo vs simulation** labeling for ``data/`` JSON.

Truthfulness:
- ``loaded: True`` only when a **recognized** JSON/GeoJSON parse succeeds — never faked.
- Judged SpectrumX detector is separate; nothing here drives official scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.edge_ran_gary.simulation_provenance import attach_execution_surface, attach_provenance_tier

# --- Primary drop zones (repo root–relative) ---
DATA_DEEPMIMO_REL = Path("data/deepmimo")
DATA_SIONNA_RT_REL = Path("data/sionna_rt")
DATA_AERIAL_OMNIVERSE_REL = Path("data/aerial_omniverse")
DATA_PYAERIAL_BRIDGE_REL = Path("data/pyaerial_bridge")
DATA_OTA_EVIDENCE_REL = Path("data/ota_evidence")
SCHEMAS_AERIAL_DATA_LAKE_REL = Path("schemas/aerial_data_lake")
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
# Written by scripts/check_ngc_access.py — never contains secret values
AERIAL_ACCESS_SUMMARY_FILE = "access_summary.json"

SIONNA_GEOJSON_NAMES = ("coverage_grid.geojson", "coverage.geojson", "sionna_coverage.geojson")

# Repo-bundled demo summaries (readable on Streamlit Cloud without writing to data/)
EXAMPLES_DEEPMIMO_REL = Path("examples/simulation_exports/deepmimo")
EXAMPLES_SIONNA_RT_REL = Path("examples/simulation_exports/sionna_rt")
EXAMPLES_AERIAL_OMNIVERSE_REL = Path("examples/simulation_exports/aerial_omniverse")
EXAMPLES_PYAERIAL_BRIDGE_REL = Path("examples/simulation_exports/pyaerial_bridge")

PYAERIAL_MANIFEST_FILES = ("bridge_manifest.json", "pyaerial_probe.json")


def _finalize_sim_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Attach data provenance + execution surface (Streamlit vs external runtime)."""
    return attach_execution_surface(attach_provenance_tier(d))


def _status_label_for_sim(loaded: bool, source_kind: str) -> str:
    if source_kind == "access":
        return "Access confirmed / installer-ready"
    if not loaded or source_kind == "absent":
        return "Not loaded"
    if source_kind == "demo":
        return "Loaded (demo summary)"
    if source_kind == "simulation":
        return "Loaded (simulation export)"
    return "Not loaded"


def _tier_source_after_provenance(tier_kind: str, raw: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """
    ``data/`` tier files without provenance count as **simulation** (operator-provided export).

    Repo export scripts should set ``export_provenance.simulation_grade``:
    - ``full_solver`` — Sionna / DeepMIMO (or equivalent) actually ran.
    - ``analytic_fallback`` / ``synthetic_template`` — downgraded to **demo** in the UI.
    """
    if tier_kind != "simulation":
        return tier_kind, None
    if not isinstance(raw, dict):
        return "simulation", None
    prov = raw.get("export_provenance")
    if not isinstance(prov, dict):
        return "simulation", None
    grade = str(prov.get("simulation_grade") or "").strip()
    if grade in ("analytic_fallback", "synthetic_template", "export_script_template"):
        return "demo", (
            "Treated as **demo summary**: `export_provenance.simulation_grade` is not `full_solver`."
        )
    if grade == "full_solver":
        return "simulation", None
    eng = str(prov.get("engine") or "").lower()
    if eng in ("sionna", "sionna_rt", "sionna_tensorflow", "sionna_jax", "deepmimo_matlab", "deepmimo_python"):
        return "simulation", None
    if eng in ("analytic", "numpy", "none", ""):
        if prov.get("script"):
            return "demo", "Treated as **demo summary**: analytic / non-solver `export_provenance.engine`."
    return "simulation", None


def _aerial_access_summary_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if not obj.get("access_summary_version"):
        return False
    checks = obj.get("checks")
    if not isinstance(checks, dict):
        return False
    return True


def _deepmimo_tiers(repo_root: Path, demo_only: bool) -> List[Tuple[List[Path], str]]:
    r = repo_root.resolve()
    sim_dirs = [p for p in (r / DATA_DEEPMIMO_REL, r / LEGACY_DEEPMIMO_REL) if p.is_dir()]
    demo_p = r / EXAMPLES_DEEPMIMO_REL
    demo_dirs = [demo_p] if demo_p.is_dir() else []
    if demo_only:
        return [(demo_dirs, "demo")] if demo_dirs else []
    out: List[Tuple[List[Path], str]] = []
    if sim_dirs:
        out.append((sim_dirs, "simulation"))
    if demo_dirs:
        out.append((demo_dirs, "demo"))
    return out


def _sionna_tiers(repo_root: Path, demo_only: bool) -> List[Tuple[List[Path], str]]:
    r = repo_root.resolve()
    sim_dirs = [p for p in (r / DATA_SIONNA_RT_REL, r / LEGACY_SIONNA_RT_REL) if p.is_dir()]
    demo_p = r / EXAMPLES_SIONNA_RT_REL
    demo_dirs = [demo_p] if demo_p.is_dir() else []
    if demo_only:
        return [(demo_dirs, "demo")] if demo_dirs else []
    out: List[Tuple[List[Path], str]] = []
    if sim_dirs:
        out.append((sim_dirs, "simulation"))
    if demo_dirs:
        out.append((demo_dirs, "demo"))
    return out


def _aerial_tiers(repo_root: Path, demo_only: bool) -> List[Tuple[Path, str]]:
    r = repo_root.resolve()
    data_d = r / DATA_AERIAL_OMNIVERSE_REL
    demo_d = r / EXAMPLES_AERIAL_OMNIVERSE_REL
    if demo_only:
        return [(demo_d, "demo")] if demo_d.is_dir() else []
    out: List[Tuple[Path, str]] = []
    if data_d.is_dir():
        out.append((data_d, "simulation"))
    if demo_d.is_dir():
        out.append((demo_d, "demo"))
    return out


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
        "pyaerial_bridge": {
            **_merge_dir_status([r / DATA_PYAERIAL_BRIDGE_REL], tuple(PYAERIAL_MANIFEST_FILES)),
            "suggested_inputs": list(PYAERIAL_MANIFEST_FILES),
            "status": "integration_ready_next_scaling",
            "note": "PHY bridge drop zone; optional pyAerial probe JSON. See docs/PYAERIAL_BRIDGE.md.",
        },
        "ota_evidence": {
            **_dir_status(r / DATA_OTA_EVIDENCE_REL),
            "suggested_inputs": ("ota_lake_manifest.json", "captures/*.json"),
            "status": "ota_ready_target",
            "note": "OTA / Data Lake target — **not active** until manifests + captures exist. See docs/DATA_LAKE_SCHEMA.md.",
        },
        "schemas_aerial_data_lake": {
            **_dir_status(r / SCHEMAS_AERIAL_DATA_LAKE_REL),
            "suggested_inputs": ("ota_capture_record.schema.json",),
            "status": "schema_stub",
            "note": "JSON Schema stubs for OTA record shape.",
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
        "data_pyaerial_bridge": {
            "path": str((root / DATA_PYAERIAL_BRIDGE_REL).resolve()),
            "exists": (root / DATA_PYAERIAL_BRIDGE_REL).is_dir(),
        },
        "data_ota_evidence": {
            "path": str((root / DATA_OTA_EVIDENCE_REL).resolve()),
            "exists": (root / DATA_OTA_EVIDENCE_REL).is_dir(),
        },
        "schemas_aerial_data_lake": {
            "path": str((root / SCHEMAS_AERIAL_DATA_LAKE_REL).resolve()),
            "exists": (root / SCHEMAS_AERIAL_DATA_LAKE_REL).is_dir(),
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


def _try_deepmimo_valid_in_dirs(
    dirs: List[Path],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    """First JSON in dirs that decodes **and** passes DeepMIMO validation."""
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
            except json.JSONDecodeError as e:
                last_err = f"JSON decode ({p.name}): {e}"
                continue
            except OSError as e:
                last_err = str(e)
                continue
            extracted = _normalize_deepmimo_dict(raw_data)
            if _deepmimo_parse_valid(extracted, raw_data):
                return raw_data, p, None
            last_err = "validation_failed"
    return None, None, last_err


def load_deepmimo_scenario_summary(repo_root: Path, demo_only: bool = False) -> Dict[str, Any]:
    """
    Load and **validate** DeepMIMO JSON summary + optional ``channel_features.npz`` metadata.
    ``loaded`` is True only if JSON parses and passes field validation.

    Priority (when ``demo_only`` is False): ``data/deepmimo/`` (and legacy path), then
    ``examples/simulation_exports/deepmimo/``. When ``demo_only`` is True, only the examples path
    is scanned (ignores ``data/``).
    """
    r = repo_root.resolve()
    tiers = _deepmimo_tiers(r, demo_only)
    primary_path = str((r / DATA_DEEPMIMO_REL).resolve())
    last_err: Optional[str] = None

    for dirs, source_kind in tiers:
        raw_data, json_path, err = _try_deepmimo_valid_in_dirs(dirs)
        last_err = err or last_err
        if raw_data is None or json_path is None:
            continue
        npz_meta = None
        parent = json_path.parent
        for cand in ("channel_features.npz", "channels.npz", "deepmimo_channels.npz"):
            m = _scan_npz_metadata(parent / cand)
            if m and "arrays" in m:
                npz_meta = m
                break
        extracted = _normalize_deepmimo_dict(raw_data)
        sk_eff, prov_note = _tier_source_after_provenance(source_kind, raw_data)
        return _finalize_sim_dict(
            {
                "loaded": True,
                "path": str(json_path.resolve()),
                "summary_json_path": str(json_path.resolve()),
                "data": raw_data,
                "extracted_summary": extracted,
                "npz_channel_meta": npz_meta,
                "integration": "deepmimo",
                "parser": "deepmimo_v3",
                "source_kind": sk_eff,
                "status_label": _status_label_for_sim(True, sk_eff),
                "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
                "parser_note": prov_note,
                "error": None,
            }
        )

    return _finalize_sim_dict(
        {
            "loaded": False,
            "path": primary_path,
            "summary_json_path": None,
            "expected_files": list(DEEPMIMO_SUMMARY_FILES),
            "expected_schema": (
                "JSON object with e.g. scenario_name, num_bs, num_users, los_links/nlos_links, "
                "or channel_statistics / sites — see docs/SIMULATION_BACKBONE_PLAN.md"
            ),
            "error": last_err,
            "integration": "deepmimo",
            "parser": "deepmimo_v3",
            "source_kind": "absent",
            "status_label": "Not loaded",
            "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
            "demo_fallback_dir": str((r / EXAMPLES_DEEPMIMO_REL).resolve()),
        }
    )


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


def _load_sionna_bundle_from_dirs(
    dirs: List[Path], source_kind: str, primary_fallback: str
) -> Dict[str, Any]:
    """Try JSON + GeoJSON only within ``dirs`` (one tier: simulation or demo)."""
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
                cand = obj if isinstance(obj, dict) else {"value": obj}
            except json.JSONDecodeError as e:
                last_err = f"JSON decode ({p.name}): {e}"
                continue
            except OSError as e:
                last_err = str(e)
                continue
            extracted_try = _normalize_sionna_dict(cand)
            if _sionna_parse_valid(extracted_try, cand):
                raw_data = cand
                json_path = p
                break
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

    parser_note: Optional[str] = None
    if not loaded:
        parser_note = "No valid Sionna summary JSON and no valid coverage GeoJSON in this search tier."
    elif not json_ok and geo_ok:
        parser_note = "Loaded from **GeoJSON only** — add propagation_summary.json for richer metrics."

    path_shown = (
        str(json_path.resolve())
        if json_path
        else (str(gj_path.resolve()) if gj_path else primary_fallback)
    )

    eff_sk = source_kind if loaded else "absent"
    prov_adj: Optional[str] = None
    if loaded and isinstance(raw_data, dict) and json_ok:
        eff_sk, prov_adj = _tier_source_after_provenance(source_kind, raw_data)
    merged_note = parser_note
    if prov_adj:
        merged_note = f"{merged_note} {prov_adj}" if merged_note else prov_adj

    return _finalize_sim_dict(
        {
            "loaded": loaded,
            "path": path_shown,
            "summary_json_path": str(json_path.resolve()) if json_path else None,
            "geojson_path": str(gj_path.resolve()) if gj_path and geo_ok else None,
            "data": raw_data if raw_data is not None else {},
            "extracted_summary": combined_extracted,
            "geojson_stats": geo_stats if geo_ok else {},
            "geojson_for_deck": geojson_inline if loaded and geo_ok else None,
            "coverage_overlay_active": bool(loaded and geo_ok and geojson_inline is not None),
            "integration": "sionna_rt",
            "parser": "sionna_rt_v3",
            "parser_note": merged_note,
            "error": None if loaded else (last_err or geo_err or "not_found"),
            "source_kind": eff_sk if loaded else "absent",
            "status_label": _status_label_for_sim(loaded, eff_sk if loaded else "absent"),
        }
    )


def load_sionna_propagation_summary(repo_root: Path, demo_only: bool = False) -> Dict[str, Any]:
    """
    Load Sionna RT **propagation_summary** / **path_loss_summary** JSON and/or **coverage_grid.geojson**.

    Priority (when ``demo_only`` is False): ``data/sionna_rt/`` (and legacy), then
    ``examples/simulation_exports/sionna_rt/``. When ``demo_only`` is True, only examples are scanned.
    """
    r = repo_root.resolve()
    primary_path = str((r / DATA_SIONNA_RT_REL).resolve())
    tiers = _sionna_tiers(r, demo_only)
    last_note: Optional[str] = None

    for dirs, source_kind in tiers:
        if not dirs:
            continue
        out = _load_sionna_bundle_from_dirs(dirs, source_kind, primary_path)
        last_note = out.get("parser_note") or last_note
        if out["loaded"]:
            out["load_mode"] = "demo_only" if demo_only else "data_first_with_demo_fallback"
            out["expected_files"] = list(SIONNA_SUMMARY_FILES) + list(SIONNA_GEOJSON_NAMES)
            out["expected_schema"] = (
                "JSON: scenario_name / path loss / LOS fraction / coverage block; "
                "or GeoJSON FeatureCollection for coverage_grid — see docs/SIMULATION_BACKBONE_PLAN.md"
            )
            return _finalize_sim_dict(out)

    return _finalize_sim_dict(
        {
            "loaded": False,
            "path": primary_path,
            "summary_json_path": None,
            "geojson_path": None,
            "data": {},
            "extracted_summary": {},
            "geojson_stats": {},
            "geojson_for_deck": None,
            "coverage_overlay_active": False,
            "integration": "sionna_rt",
            "parser": "sionna_rt_v3",
            "parser_note": last_note or "No valid Sionna summary JSON and no valid coverage GeoJSON found.",
            "error": "not_found",
            "expected_files": list(SIONNA_SUMMARY_FILES) + list(SIONNA_GEOJSON_NAMES),
            "expected_schema": (
                "JSON: scenario_name / path loss / LOS fraction / coverage block; "
                "or GeoJSON FeatureCollection for coverage_grid — see docs/SIMULATION_BACKBONE_PLAN.md"
            ),
            "source_kind": "absent",
            "status_label": "Not loaded",
            "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
            "demo_fallback_dir": str((r / EXAMPLES_SIONNA_RT_REL).resolve()),
        }
    )


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


def _merge_aerial_access_and_tier(hit: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """Attach ``access_summary.json`` (if present) and compute ``aerial_status_tier`` / headline status."""
    acc_path = repo_root / DATA_AERIAL_OMNIVERSE_REL / AERIAL_ACCESS_SUMMARY_FILE
    hit["manifest_loaded"] = bool(hit.get("loaded"))
    hit["access_confirmed"] = False
    hit["access_summary_path"] = None
    hit["access_summary_excerpt"] = {}
    if acc_path.is_file():
        try:
            ad = json.loads(acc_path.read_text(encoding="utf-8"))
            if _aerial_access_summary_valid(ad):
                hit["access_confirmed"] = True
                hit["access_summary_path"] = str(acc_path.resolve())
                hit["access_summary_excerpt"] = {
                    "access_summary_version": ad.get("access_summary_version"),
                    "generated_at": ad.get("generated_at"),
                    "checks": ad.get("checks"),
                    "env_presence": ad.get("env_presence"),
                }
        except (json.JSONDecodeError, OSError):
            pass

    if hit.get("loaded"):
        hit["aerial_status_tier"] = "manifest"
    elif hit.get("access_confirmed"):
        hit["aerial_status_tier"] = "access_confirmed"
        hit["status_label"] = "Access confirmed / installer-ready"
        hit["source_kind"] = "access"
    else:
        hit["aerial_status_tier"] = "none"
        if not hit.get("status_label"):
            hit["status_label"] = "Not loaded"
        if hit.get("source_kind") is None:
            hit["source_kind"] = "absent"
    return hit


def _try_aerial_in_dir(d: Path, source_kind: str) -> Optional[Dict[str, Any]]:
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

    if raw_data is None or json_path is None:
        return None

    extracted = _normalize_aerial_dict(raw_data)
    ok = _aerial_parse_valid(extracted, raw_data)
    if not ok:
        return {
            "loaded": False,
            "path": str(json_path.resolve()),
            "data": raw_data,
            "extracted_summary": extracted,
            "integration": "aerial_omniverse",
            "parser": "aerial_v3",
            "external_tooling_required": True,
            "parser_note": "JSON present but missing expected manifest fields — not marked loaded.",
            "error": "validation_failed",
            "source_kind": "absent",
            "status_label": "Not loaded",
            "summary_json_path": str(json_path.resolve()),
        }

    return {
        "loaded": True,
        "path": str(json_path.resolve()),
        "summary_json_path": str(json_path.resolve()),
        "data": raw_data,
        "extracted_summary": extracted,
        "integration": "aerial_omniverse",
        "parser": "aerial_v3",
        "external_tooling_required": True,
        "parser_note": None,
        "error": None,
        "source_kind": source_kind,
        "status_label": _status_label_for_sim(True, source_kind),
    }


def load_aerial_overlay_summary(repo_root: Path, demo_only: bool = False) -> Dict[str, Any]:
    """
    Load twin_manifest / overlay_summary JSON only if it passes validation.

    Priority: ``data/aerial_omniverse/`` then ``examples/simulation_exports/aerial_omniverse/``
    unless ``demo_only`` is True (examples only).

    Also reads ``data/aerial_omniverse/access_summary.json`` (from ``check_ngc_access``) to surface
    **Access confirmed / installer-ready** when no manifest is loaded — **without** implying a twin export.
    """
    r = repo_root.resolve()
    tiers = _aerial_tiers(r, demo_only)
    default_path = str(aerial_drop_dir(r))
    last_err: Optional[str] = None

    for d, source_kind in tiers:
        hit = _try_aerial_in_dir(d, source_kind)
        if hit is None:
            continue
        if hit["loaded"]:
            hit["load_mode"] = "demo_only" if demo_only else "data_first_with_demo_fallback"
            return _finalize_sim_dict(_merge_aerial_access_and_tier(hit, r))
        last_err = hit.get("error")

    base = {
        "loaded": False,
        "path": default_path,
        "summary_json_path": None,
        "expected_files": list(AERIAL_SUMMARY_FILES) + [AERIAL_ACCESS_SUMMARY_FILE],
        "expected_schema": (
            "Manifest: scene_name / usd_path or aerial_export_version — "
            "**NVIDIA AI Aerial / Omniverse** runs **outside** this repo. "
            "Optional: access_summary.json from scripts/check_ngc_access.py (no secrets)."
        ),
        "integration": "aerial_omniverse",
        "parser": "aerial_v3",
        "external_tooling_required": True,
        "error": last_err,
        "source_kind": "absent",
        "status_label": "Not loaded",
        "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
        "demo_fallback_dir": str((r / EXAMPLES_AERIAL_OMNIVERSE_REL).resolve()),
        "data": {},
        "extracted_summary": {},
    }
    return _finalize_sim_dict(_merge_aerial_access_and_tier(base, r))


def _pyaerial_tiers(repo_root: Path, demo_only: bool) -> List[Tuple[Path, str]]:
    r = repo_root.resolve()
    out: List[Tuple[Path, str]] = []
    if not demo_only and (r / DATA_PYAERIAL_BRIDGE_REL).is_dir():
        out.append((r / DATA_PYAERIAL_BRIDGE_REL, "simulation"))
    ex = r / EXAMPLES_PYAERIAL_BRIDGE_REL
    if ex.is_dir():
        out.append((ex, "demo"))
    return out


def load_pyaerial_bridge_status(repo_root: Path, demo_only: bool = False) -> Dict[str, Any]:
    """
    Load optional ``bridge_manifest.json`` / ``pyaerial_probe.json`` from ``data/pyaerial_bridge/``
    or bundled ``examples/simulation_exports/pyaerial_bridge/``.

    Does **not** execute pyAerial; surfaces ``phy_env`` import probe from ``phy_interface``.
    """
    from src.edge_ran_gary.pyaerial_bridge.phy_interface import describe_pyaerial_environment

    phy = describe_pyaerial_environment()
    r = repo_root.resolve()
    last_err: Optional[str] = None
    primary = str((r / DATA_PYAERIAL_BRIDGE_REL).resolve())

    for d, source_kind in _pyaerial_tiers(r, demo_only):
        if not d.is_dir():
            continue
        for fn in PYAERIAL_MANIFEST_FILES:
            p = d / fn
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                raw = obj if isinstance(obj, dict) else {"value": obj}
            except (json.JSONDecodeError, OSError) as e:
                last_err = str(e)
                continue
            if not raw.get("bridge_manifest_version") and not raw.get("pyaerial_probe_version"):
                last_err = "validation_failed"
                continue
            sk_eff, prov_note = _tier_source_after_provenance(source_kind, raw)
            extracted = {k: raw.get(k) for k in ("status", "notes", "generated_at", "bridge_manifest_version") if raw.get(k) is not None}
            return _finalize_sim_dict(
                {
                    "loaded": True,
                    "path": str(p.resolve()),
                    "summary_json_path": str(p.resolve()),
                    "data": raw,
                    "extracted_summary": extracted,
                    "integration": "pyaerial_bridge",
                    "parser": "pyaerial_bridge_v1",
                    "source_kind": sk_eff,
                    "status_label": _status_label_for_sim(True, sk_eff),
                    "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
                    "parser_note": prov_note,
                    "phy_env": {"pyaerial_import_ok": phy.pyaerial_import_ok, "import_error": phy.import_error},
                    "error": None,
                }
            )

    return _finalize_sim_dict(
        {
            "loaded": False,
            "path": primary,
            "integration": "pyaerial_bridge",
            "parser": "pyaerial_bridge_v1",
            "source_kind": "absent",
            "status_label": "Not loaded",
            "load_mode": "demo_only" if demo_only else "data_first_with_demo_fallback",
            "expected_files": list(PYAERIAL_MANIFEST_FILES),
            "phy_env": {"pyaerial_import_ok": phy.pyaerial_import_ok, "import_error": phy.import_error},
            "error": last_err,
            "demo_fallback_dir": str((r / EXAMPLES_PYAERIAL_BRIDGE_REL).resolve()),
        }
    )


def load_ota_evidence_status(repo_root: Path) -> Dict[str, Any]:
    """OTA / Data Lake manifest under ``data/ota_evidence/`` (see ``ota_data_interface``)."""
    from src.edge_ran_gary.ota_data_interface import load_ota_lake_manifest

    return _finalize_sim_dict(load_ota_lake_manifest(repo_root))


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

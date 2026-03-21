"""
Load and run SpectrumX-DAC-style submission packages from ``submissions/<name>/main.py``.

Each package must expose ``evaluate(filename: str) -> int`` (0/1). Optional tuple returns
``(pred, confidence)`` are supported for Streamlit display.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Strongest-known packages first (adjust if you add newer folders).
PREFERRED_SUBMISSION_ORDER: List[str] = [
    "leaderboard_v9",
    "leaderboard_v12",
    "leaderboard_v11",
    "leaderboard_v10",
    "leaderboard_v5",
    "leaderboard_baseline_v1",
]

# Do not scan deeper than this under submissions/ (avoids venvs / accidents).
_MAX_SUBMISSION_DEPTH = 4


def resolve_repo_root(anchor: Optional[Path] = None) -> Path:
    """
    Resolve the repository root without relying on the process cwd.

    Walks upward from ``anchor`` (default: this file) looking for ``submissions/``
    or ``src/edge_ran_gary``. Falls back to three parents of this file
    (``src/edge_ran_gary/submission_adapter.py`` -> repo root).
    """
    start = (anchor or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        subs = p / "submissions"
        if subs.is_dir():
            return p
        if (p / "src" / "edge_ran_gary").is_dir():
            return p
    # Default: .../repo/src/edge_ran_gary/submission_adapter.py -> parents[2] == repo
    return Path(__file__).resolve().parents[2]


def submission_package_path(repo_root: Path, folder_name: str) -> Path:
    """
    Path to ``submissions/<folder_name>/`` where *folder_name* may be nested
    (e.g. ``leaderboard_baseline_v1/test3k``). Safe for POSIX paths; normalizes
    backslashes to slashes.
    """
    fn = (folder_name or "").replace("\\", "/").strip("/")
    if not fn:
        return (repo_root / "submissions" / "_invalid").resolve()
    return repo_root.joinpath("submissions", *fn.split("/")).resolve()


def discover_submission_folders(repo_root: Path) -> List[str]:
    """
    Discover packages under ``submissions/**/main.py`` up to depth
    ``_MAX_SUBMISSION_DEPTH`` (relative to ``submissions``), plus stable sort
    with ``PREFERRED_SUBMISSION_ORDER`` applied to the *top-level* segment.
    """
    sub = (repo_root / "submissions").resolve()
    if not sub.is_dir():
        return []
    found: set[str] = set()
    try:
        for main_py in sub.rglob("main.py"):
            if "__pycache__" in main_py.parts:
                continue
            try:
                rel_parent = main_py.parent.relative_to(sub)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel_parent.parts):
                continue
            if len(rel_parent.parts) > _MAX_SUBMISSION_DEPTH:
                continue
            found.add(rel_parent.as_posix())
    except OSError:
        return []

    def sort_key(name: str) -> Tuple[int, str]:
        top = name.split("/")[0]
        if top in PREFERRED_SUBMISSION_ORDER:
            return (PREFERRED_SUBMISSION_ORDER.index(top), name)
        return (len(PREFERRED_SUBMISSION_ORDER), name)

    return sorted(found, key=sort_key)


def default_best_submission_folder(repo_root: Path) -> Optional[str]:
    folders = discover_submission_folders(repo_root)
    return folders[0] if folders else None


def load_submission_module(submission_dir: Path) -> Any:
    """Import submissions/<pkg>/main.py as a one-off module."""
    main_py = submission_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"Missing main.py: {main_py}")
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", submission_dir.name)
    spec = importlib.util.spec_from_file_location(
        f"submission_pkg_{safe_name}",
        str(main_py),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {main_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_evaluate_fn(module: Any) -> Callable[..., Any]:
    fn = getattr(module, "evaluate", None)
    if fn is None or not callable(fn):
        raise AttributeError("main.py must define evaluate(filename: str)")
    return fn


def run_evaluate_on_iq_array(
    module: Any,
    iq: np.ndarray,
) -> Tuple[int, Dict[str, Any]]:
    """
    Write IQ to a temp .npy and call evaluate(path).

    Returns:
        (prediction 0/1, info dict with optional keys: confidence, error, raw_type)
    """
    info: Dict[str, Any] = {
        "confidence": None,
        "error": None,
        "raw_type": None,
        "fallback_active": False,
        "trained_path_active": False,
    }
    evaluate_fn = get_evaluate_fn(module)
    fd, path = tempfile.mkstemp(suffix=".npy", prefix="streamlit_iq_")
    os.close(fd)
    try:
        np.save(path, iq.astype(np.complex64))
        raw = evaluate_fn(path)
        info["raw_type"] = type(raw).__name__
        if isinstance(raw, (tuple, list)) and len(raw) >= 1:
            pred = int(raw[0])
            if len(raw) >= 2 and raw[1] is not None:
                try:
                    info["confidence"] = float(raw[1])
                except (TypeError, ValueError):
                    info["confidence"] = str(raw[1])
        else:
            pred = int(raw)
        # Heuristic: inspect module text for fallback wording (display only).
        main_py = Path(module.__file__) if getattr(module, "__file__", None) else None
        if main_py and main_py.is_file():
            txt = main_py.read_text(encoding="utf-8", errors="ignore").lower()
            info["trained_path_active"] = any(
                k in txt for k in (".npz", "load(", "joblib", "pickle", "artifact")
            )
            info["fallback_active"] = any(
                k in txt for k in ("fallback", "else:", "energy", "flatness", "baseline")
            )
        return pred, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return 0, info
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def submission_folder_info(repo_root: Path, folder_name: str) -> Dict[str, Any]:
    """Lightweight read-only summary for UI (no execution)."""
    d = submission_package_path(repo_root, folder_name)
    out: Dict[str, Any] = {
        "folder": folder_name,
        "main_py": (d / "main.py").is_file(),
        "user_reqs": (d / "user_reqs.txt").is_file(),
        "artifact_present": False,
        "artifacts": [],
        "model_family_guess": "unknown",
        "default_sample_rate_hz": None,
    }
    if not d.is_dir():
        return out
    arts: List[str] = []
    for pat in ("*.npz", "*.pkl", "*.joblib"):
        arts.extend([a.name for a in d.glob(pat)])
    out["artifacts"] = sorted(set(arts))[:20]
    out["artifact_present"] = len(out["artifacts"]) > 0
    main_py = d / "main.py"
    if main_py.is_file():
        txt = main_py.read_text(encoding="utf-8", errors="ignore")[:120000]
        nl = txt.lower()
        if "logistic" in nl or "linearsvc" in nl or "svm" in nl:
            out["model_family_guess"] = "feature_linear"
        elif "spectral" in nl and "flatness" in nl:
            out["model_family_guess"] = "spectral_flatness_heuristic"
        elif "energy" in nl:
            out["model_family_guess"] = "energy_heuristic"
        m = re.search(r"DEFAULT_SAMPLE_RATE\s*=\s*([0-9.eE+-]+)", txt)
        if m:
            try:
                out["default_sample_rate_hz"] = float(m.group(1))
            except ValueError:
                pass
    return out

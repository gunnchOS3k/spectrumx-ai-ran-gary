"""
SpX-DAC Baseline Comparison Dashboard

Streamlit dashboard for visualizing IQ data and running baseline detection models.
Supports user-uploaded .npy files with various IQ formats.
"""

import streamlit as st
import io
from pathlib import Path

# Smoke check: optional heavy deps — show friendly message instead of crash
try:
    import numpy as np
    from scipy import signal
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore
except ImportError as e:
    st.error("Missing dependencies for visualizations.")
    st.code(str(e))
    st.info("Run: `pip install -r requirements.txt`")
    st.stop()

# Optional: pydeck for interactive 3D scenes (Future Work / Micro-Twin)
try:
    import pydeck as pdk  # type: ignore
except Exception:
    pdk = None  # type: ignore

# Page config
st.set_page_config(
    page_title="SpectrumX DAC — Project Dashboard",
    page_icon="📡",
    layout="wide",
)

# Title
st.title("📡 SpectrumX DAC — Winning Project Dashboard")
st.caption(
    "Judge Mode highlights the **core SpectrumX DAC detector** (local metrics + live `evaluate()` on synthetic IQ). "
    "**Future work** (Micro-Twin, 6G research path) is clearly separated."
)

# Safety banner: do not upload competition data to Cloud
st.warning(
    "**Do NOT upload official competition IQ data to Streamlit Cloud.** "
    "Use synthetic micro-twin data or run locally for real datasets."
)

# Optional shared feature extractor (for report figures)
try:
    from src.edge_ran_gary.detection.feature_baseline import extract_features as shared_extract_features
except Exception:
    shared_extract_features = None

# Leaderboard submission packages: load submissions/<pkg>/main.py and call evaluate()
try:
    from src.edge_ran_gary.submission_adapter import (
        discover_submission_folders as _sa_discover_submission_folders,
        default_best_submission_folder as _sa_default_best_submission_folder,
        load_submission_module as _sa_load_submission_module,
        run_evaluate_on_iq_array as _sa_run_evaluate_on_iq_array,
        submission_folder_info as _sa_submission_folder_info,
    )

    _SUBMISSION_ADAPTER_OK = True
except Exception:
    _SUBMISSION_ADAPTER_OK = False
    _sa_discover_submission_folders = None  # type: ignore
    _sa_default_best_submission_folder = None  # type: ignore
    _sa_load_submission_module = None  # type: ignore
    _sa_run_evaluate_on_iq_array = None  # type: ignore
    _sa_submission_folder_info = None  # type: ignore

SUBMISSION_MODEL_FINAL = "Final Submission (Best Known)"
SUBMISSION_MODEL_EXPLORER = "Submission Explorer"


def _discover_submissions_safe(repo_root: Path) -> list:
    if not _SUBMISSION_ADAPTER_OK or _sa_discover_submission_folders is None:
        return []
    try:
        return list(_sa_discover_submission_folders(repo_root))
    except Exception:
        return []


def _default_best_pkg(repo_root: Path) -> str | None:
    if not _SUBMISSION_ADAPTER_OK or _sa_default_best_submission_folder is None:
        return None
    try:
        return _sa_default_best_submission_folder(repo_root)
    except Exception:
        return None


@st.cache_resource
def _cached_submission_module(submission_dir_abs: str):
    """Load submissions/<pkg>/main.py once per process (folder absolute path)."""
    if _sa_load_submission_module is None:
        raise RuntimeError("submission adapter not available")
    return _sa_load_submission_module(Path(submission_dir_abs))


# ============================================================================
# Helper Functions
# ============================================================================

def load_iq_data(uploaded_file, is_int16_interleaved=False):
    """
    Load IQ data from .npy file with support for multiple formats.
    
    Supported formats:
    - complex array shape (N,) dtype complex64/complex128
    - float array shape (N,2) interpreted as [I,Q]
    - int16 interleaved IQ (N*2,) when is_int16_interleaved=True
    
    Returns:
        complex64 array of shape (N,)
    """
    try:
        data = np.load(io.BytesIO(uploaded_file.read()), allow_pickle=False)
        
        # Reset file pointer for potential re-read
        uploaded_file.seek(0)
        
        if is_int16_interleaved:
            # int16 interleaved: [I0, Q0, I1, Q1, ...]
            if data.dtype != np.int16:
                st.warning(f"Expected int16 for interleaved format, got {data.dtype}")
            if len(data.shape) != 1 or data.shape[0] % 2 != 0:
                st.error("Interleaved IQ data must be 1D with even length")
                return None
            i = data[::2].astype(np.float32)
            q = data[1::2].astype(np.float32)
            iq_complex = i + 1j * q
            return iq_complex.astype(np.complex64)
        
        # Check if already complex
        if np.iscomplexobj(data):
            if len(data.shape) == 1:
                return data.astype(np.complex64)
            else:
                st.error(f"Complex data must be 1D, got shape {data.shape}")
                return None
        
        # Check if float array with shape (N, 2)
        if len(data.shape) == 2 and data.shape[1] == 2:
            i = data[:, 0].astype(np.float32)
            q = data[:, 1].astype(np.float32)
            iq_complex = i + 1j * q
            return iq_complex.astype(np.complex64)
        
        # Check if 1D float (try to interpret as I-only, warn)
        if len(data.shape) == 1:
            st.warning("1D float array detected. Assuming this is I component only (Q=0).")
            return data.astype(np.complex64)
        
        st.error(f"Unsupported data shape: {data.shape}, dtype: {data.dtype}")
        return None
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


@st.cache_data
def compute_psd(iq_data, sample_rate, nperseg=1024):
    """Compute PSD using Welch's method."""
    freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=nperseg, 
                              return_onesided=False, scaling='density')
    return freqs, psd


@st.cache_data
def compute_spectrogram(iq_data, sample_rate, nperseg=256, noverlap=None):
    """Compute spectrogram using STFT."""
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, times, Sxx = signal.spectrogram(
        iq_data, fs=sample_rate, nperseg=nperseg, 
        noverlap=noverlap, return_onesided=False, scaling='density'
    )
    return freqs, times, Sxx


def energy_detector(iq_data, threshold):
    """
    Simple energy detector baseline.
    
    Args:
        iq_data: complex64 array
        threshold: power threshold
        
    Returns:
        prediction (0/1), confidence (0-1)
    """
    power = np.mean(np.abs(iq_data) ** 2)
    prediction = 1 if power > threshold else 0
    # Confidence based on distance from threshold (normalized)
    distance = abs(power - threshold) / (threshold + 1e-10)
    confidence = min(1.0, distance)
    return prediction, confidence, power


def spectral_flatness_detector(iq_data, sample_rate, threshold):
    """
    Spectral flatness detector baseline.
    
    Spectral flatness = geometric_mean(PSD) / arithmetic_mean(PSD)
    Lower flatness indicates more structured signal (less noise-like).
    
    Args:
        iq_data: complex64 array
        sample_rate: sample rate in Hz
        threshold: flatness threshold
        
    Returns:
        prediction (0/1), confidence (0-1)
    """
    freqs, psd = compute_psd(iq_data, sample_rate)
    # Use magnitude of PSD (handle negative freqs)
    psd_mag = np.abs(psd)
    psd_mag = psd_mag[psd_mag > 0]  # Avoid log(0)
    
    if len(psd_mag) == 0:
        return 0, 0.0, 0.0
    
    geometric_mean = np.exp(np.mean(np.log(psd_mag)))
    arithmetic_mean = np.mean(psd_mag)
    
    if arithmetic_mean == 0:
        flatness = 0.0
    else:
        flatness = geometric_mean / arithmetic_mean
    
    # Lower flatness = more signal-like (prediction=1)
    # Higher flatness = more noise-like (prediction=0)
    prediction = 1 if flatness < threshold else 0
    distance = abs(flatness - threshold) / (threshold + 1e-10)
    confidence = min(1.0, distance)
    
    return prediction, confidence, flatness


def psd_logreg_detector(iq_data, sample_rate):
    """
    Placeholder for PSD+LogReg model.
    Returns placeholder if model not available.
    """
    # TODO: Implement if model exists in src/
    return None


# ----------------------------------------------------------------------------
# Report / Figure Mode helpers
# ----------------------------------------------------------------------------

def _ensure_session_defaults():
    st.session_state.setdefault("ui_mode", "Standard Mode")
    st.session_state.setdefault("figure_screenshot_preset", True)
    st.session_state.setdefault("figure_show_captions", True)
    st.session_state.setdefault("report_section_title", "Experimental Results")
    st.session_state.setdefault("report_figure_number", "Figure X")
    st.session_state.setdefault(
        "report_caption_text",
        "Synthetic/demo IQ shown for illustration. Do not upload official competition data to Streamlit Cloud.",
    )
    # Headline numbers (editable for screenshot staging)
    st.session_state.setdefault("results_baseline_name", "Spectral Flatness (baseline)")
    st.session_state.setdefault("results_baseline_metric", "TBD")
    st.session_state.setdefault("results_improved_name", "Feature model (trained, local)")
    st.session_state.setdefault("results_improved_metric", "TBD")


def _caption(text: str):
    if st.session_state.get("ui_mode") == "Figure Mode" and st.session_state.get("figure_show_captions", True):
        st.caption(text)


def _plot_heights():
    preset = st.session_state.get("figure_screenshot_preset", True)
    if st.session_state.get("ui_mode") != "Figure Mode":
        return {"time": 600, "constellation": 500, "psd": 400, "spec": 500}
    if preset:
        return {"time": 750, "constellation": 600, "psd": 520, "spec": 650}
    return {"time": 650, "constellation": 520, "psd": 440, "spec": 560}


def _features_dataframe(iq: np.ndarray, sample_rate: float):
    if shared_extract_features is None:
        return None
    try:
        feats = shared_extract_features(iq, sample_rate=float(sample_rate))
        if not feats:
            return None
        rows = [{"feature": k, "value": float(v)} for k, v in sorted(feats.items())]
        return rows
    except Exception:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@st.cache_data
def scan_submissions_inventory(repo_root_str: str) -> list:
    """Read-only inventory of submissions/* folders (no competition data)."""
    root = Path(repo_root_str)
    sub_root = root / "submissions"
    rows: list = []
    if not sub_root.is_dir():
        return rows
    for p in sorted(sub_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        main_py = p / "main.py"
        req = p / "user_reqs.txt"
        artifacts: list = []
        for pat in ("*.npz", "*.pkl", "*.joblib"):
            try:
                artifacts.extend(p.glob(pat))
            except Exception:
                pass
        artifact_present = len(artifacts) > 0
        artifact_names = ", ".join(sorted({a.name for a in artifacts}))[:500]

        sig: dict = {
            "mentions_spectral_flatness": False,
            "mentions_energy_detector": False,
            "mentions_extract_features": False,
            "mentions_lr_or_svm": False,
            "mentions_trained_artifact": False,
        }
        if main_py.is_file():
            try:
                txt = main_py.read_text(encoding="utf-8", errors="ignore")[:80000]
                nl = txt.lower()
                sig["mentions_spectral_flatness"] = (
                    "spectral flatness" in nl
                    or "spectralflatness" in nl.replace(" ", "")
                    or "spectral_flatness" in nl
                )
                sig["mentions_energy_detector"] = "energy" in nl and "detector" in nl
                sig["mentions_extract_features"] = (
                    "extract_features" in txt or "extract features" in nl
                )
                sig["mentions_lr_or_svm"] = any(
                    k in nl for k in ("logistic", "linearsvc", "linear svc", " svm", "sklearn")
                )
                sig["mentions_trained_artifact"] = ".npz" in nl and (
                    "load" in nl or "weights" in nl or "artifact" in nl
                )
            except Exception:
                pass

        rows.append(
            {
                "folder": name,
                "main_py": bool(main_py.is_file()),
                "user_reqs_txt": bool(req.is_file()),
                "artifact_present": artifact_present,
                "artifacts": artifact_names,
                **sig,
            }
        )
    return rows


@st.cache_data
def load_submission_metrics(repo_root_str: str):
    """Load authoritative CV / leaderboard table if present."""
    path = Path(repo_root_str) / "submissions" / "submission_metrics.csv"
    if not path.is_file():
        return None
    if pd is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_leaderboard_summary(repo_root_str: str):
    """Load optional leaderboard summary if present."""
    path = Path(repo_root_str) / "submissions" / "leaderboard_summary.csv"
    if not path.is_file():
        return None
    if pd is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_final_report_figures_yaml(repo_root_str: str):
    """
    Optional report figure captions / titles from docs/final_report_figures.yaml.
    If missing or invalid, returns None (UI shows expected schema).
    """
    path = Path(repo_root_str) / "docs" / "final_report_figures.yaml"
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return {"_error": "PyYAML not installed; run `pip install pyyaml` to load figure captions from YAML."}
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(txt)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fig_yaml_caption(fig_yaml: dict | None, figure_key: str, default: str) -> str:
    """Resolve caption string for a figure key from optional YAML."""
    if not fig_yaml or not isinstance(fig_yaml, dict):
        return default
    if fig_yaml.get("_error"):
        return default
    root = fig_yaml.get("figures")
    if root is None:
        root = fig_yaml
    if not isinstance(root, dict):
        return default
    entry = root.get(figure_key)
    if entry is None:
        return default
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return str(entry.get("caption") or entry.get("text") or default)
    return default


def _leaderboard_progress_dataframe(mdf):
    """
    Narrow metrics to leaderboard-progress columns when present.
    Maps flexible column names (submission_version, change, etc.).
    """
    if mdf is None or pd is None:
        return None
    lower = {c.lower(): c for c in mdf.columns}

    def pick(*candidates):
        for name in candidates:
            if name in lower:
                return lower[name]
        return None

    col_sub = pick("submission", "folder", "name")
    col_ver = pick("submission_version", "version", "submission_tag", "tag")
    col_rank = pick("leaderboard_rank", "rank", "lb_rank")
    col_acc = pick("leaderboard_accuracy", "accuracy", "lb_accuracy", "score")
    col_note = pick("notes", "change", "changelog", "note", "delta")

    if col_rank is None:
        return None

    out_cols = []
    rename = {}
    if col_ver:
        out_cols.append(col_ver)
        rename[col_ver] = "submission_version"
    elif col_sub:
        out_cols.append(col_sub)
        rename[col_sub] = "submission"
    else:
        return None

    out_cols.append(col_rank)
    rename[col_rank] = "leaderboard_rank"

    if col_acc:
        out_cols.append(col_acc)
        rename[col_acc] = "leaderboard_accuracy"
    if col_note:
        out_cols.append(col_note)
        rename[col_note] = "notes_or_change"

    try:
        slim = mdf[[c for c in out_cols if c in mdf.columns]].copy()
        slim = slim.rename(columns={k: v for k, v in rename.items() if k in slim.columns})
        return slim.sort_values("leaderboard_rank", na_position="last")
    except Exception:
        return None


def _user_reqs_line_count(repo_root: Path, submission_folder: str) -> int | None:
    """Rough dependency footprint: count non-empty lines in user_reqs.txt."""
    p = repo_root / "submissions" / submission_folder / "user_reqs.txt"
    if not p.is_file():
        return None
    try:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
        return len([ln for ln in lines if ln and not ln.startswith("#")])
    except Exception:
        return None


def _interpret_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _safe_read_text(path: Path, max_chars: int = 80000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _pick_core_submission_row(metrics_df, summary_df, inventory_rows: list[dict]) -> dict | None:
    """
    Pick the "core judged submission" row from local structured metrics.
    Heuristics are strictly local; no official competition data is loaded.
    """
    # Helper: try to choose the best-looking DAC/SpectrumX row by name first.
    def _pick_from_df(df):
        if df is None:
            return None
        cols = set(df.columns)
        name_col = "submission" if "submission" in cols else None
        rank_col = "leaderboard_rank" if "leaderboard_rank" in cols else None
        acc_col = "leaderboard_accuracy" if "leaderboard_accuracy" in cols else None

        if name_col is not None:
            nl = df[name_col].astype(str).str.lower()
            for needle in ("spectrumx", "spectrum", "dac", "core"):
                mask = nl.str.contains(needle, na=False)
                if mask.any():
                    cand = df.loc[mask].copy()
                    if rank_col is not None:
                        cand = cand.sort_values(rank_col, na_position="last")
                        return cand.iloc[0].to_dict()
                    if acc_col is not None:
                        cand = cand.sort_values(acc_col, ascending=False)
                        return cand.iloc[0].to_dict()
                    return cand.iloc[0].to_dict()

        # Fallback ordering: best rank if present, else best accuracy.
        if rank_col is not None:
            try:
                return df.sort_values(rank_col, na_position="last").iloc[0].to_dict()
            except Exception:
                pass
        if acc_col is not None:
            try:
                return df.sort_values(acc_col, ascending=False).iloc[0].to_dict()
            except Exception:
                pass
        # Last resort: first row.
        try:
            return df.iloc[0].to_dict()
        except Exception:
            return None

    core = _pick_from_df(metrics_df)
    if core is not None:
        return core

    core = _pick_from_df(summary_df)
    if core is not None:
        return core

    # If we can't load structured metrics, attempt to select a plausible "core" by inventory name.
    for inv in inventory_rows:
        fn = (inv.get("folder") or "").lower()
        if any(k in fn for k in ("dac", "spectrumx", "spectrum", "core")):
            return {"submission": inv.get("folder"), "artifact_present": inv.get("artifact_present")}

    return None


def _compute_submission_artifact_footprint(repo_root: Path, submission_folder_name: str) -> tuple[float, str]:
    """Sum local learned artifact sizes for a submission folder (read-only)."""
    sub_root = repo_root / "submissions" / submission_folder_name
    if not sub_root.is_dir():
        return 0.0, "0 KB"
    total_bytes = 0
    matches: list[str] = []
    for pat in ("*.npz", "*.pkl", "*.joblib", "*.pth", "*.pt", "*.ckpt"):
        try:
            for fp in sub_root.glob(pat):
                if fp.is_file():
                    total_bytes += fp.stat().st_size
                    matches.append(fp.name)
        except Exception:
            pass
    if total_bytes <= 0:
        return 0.0, "0 KB"
    mb = total_bytes / (1024 * 1024)
    uniq = ", ".join(sorted({m for m in matches}))[:200]
    return mb, uniq


def _infer_inference_path(repo_root: Path, submission_folder_name: str) -> str:
    """Lightweight heuristic: trained artifact loads vs fallback baseline."""
    main_path = repo_root / "submissions" / submission_folder_name / "main.py"
    if not main_path.is_file():
        return "inference path unknown (missing main.py)"
    txt = _safe_read_text(main_path).lower()

    # Trained artifact indicators.
    if any(k in txt for k in ("joblib.load", "pickle.load", "torch.load", "load_model", ".npz", ".pkl", ".joblib")):
        if any(k in txt for k in ("spectral flatness", "energy_detector", "energy detector", "fallback")):
            return "hybrid: trained-artifact primary with baseline fallback"
        return "trained-artifact primary (loads a persisted model/weights)"

    # Baseline indicators.
    if "energy" in txt or "flatness" in txt:
        return "baseline fallback (energy/flatness-style detector)"

    return "inference path unknown (no clear artifact/baseline markers found)"


def _generate_synthetic_demo_iq(sample_rate: float = 1e6, duration: float = 1.0, seed: int = 42):
    """
    Generate demo/synthetic IQ for visualization only (never used as judged official input).

    Always injects a **structured carrier burst** in the **middle ~30%** of the window on top of
    complex Gaussian noise → demo class is **mixed** (not pure noise-only).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    signal_freq = 100e3
    signal_samples = int(0.3 * n_samples)
    signal_start = n_samples // 2 - signal_samples // 2
    t_burst_start = float(signal_start / sample_rate)
    t_burst_end = float((signal_start + signal_samples) / sample_rate)
    noise = rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)
    signal_phase = 2 * np.pi * signal_freq * t[signal_start : signal_start + signal_samples]
    burst = 0.5 * np.exp(1j * signal_phase)
    demo_iq = noise.astype(np.complex64)
    demo_iq[signal_start : signal_start + signal_samples] = (
        demo_iq[signal_start : signal_start + signal_samples] + burst.astype(np.complex64)
    )
    meta = {
        "demo_class": "mixed",
        "signal_inserted": True,
        "burst_fraction_of_window": 0.3,
        "burst_time_start_s": t_burst_start,
        "burst_time_end_s": t_burst_end,
        "burst_duration_s": t_burst_end - t_burst_start,
        "carrier_hz_approx": signal_freq,
        "generator_type": "Streamlit synthetic demo (Gaussian noise + middle-window phase-modulated burst)",
        "label_meaning": "Illustration only; not an official SpectrumX label.",
    }
    return demo_iq, meta


def _metrics_row_for_submission(mdf, folder_name: str | None) -> dict | None:
    """Return first CSV row whose submission/folder/name column matches the package folder."""
    if mdf is None or pd is None or not folder_name:
        return None
    for col in ("submission", "folder", "name"):
        if col in mdf.columns:
            mask = mdf[col].astype(str) == str(folder_name)
            if mask.any():
                try:
                    return mdf.loc[mask].iloc[0].to_dict()
                except Exception:
                    return None
    return None


def _render_synthetic_demo_metadata_callout(meta: dict | None, caption_prefix: str = ""):
    """UI block: explicit synthetic demo labeling (Judge / Standard / Figure)."""
    if not meta:
        return
    st.markdown("**Synthetic demo IQ — what this sample is**")
    rows = [
        {"Field": "Demo class", "Value": meta.get("demo_class", "—")},
        {"Field": "Signal inserted", "Value": "yes" if meta.get("signal_inserted") else "no"},
        {
            "Field": "Burst interval (s)",
            "Value": f"{meta.get('burst_time_start_s', '—'):.4f} – {meta.get('burst_time_end_s', '—'):.4f}",
        },
        {"Field": "Burst duration (s)", "Value": f"{meta.get('burst_duration_s', 0):.4f}"},
        {"Field": "Generator type", "Value": str(meta.get("generator_type", "—"))},
    ]
    if pd is not None:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.json(meta)
    st.info(
        (caption_prefix + " ").strip()
        + "**How to interpret:** The waveform always includes **noise everywhere** and a **middle burst** of structured energy. "
        "Do not treat quiet-looking edges as “official noise-only ground truth” — this is a **demo generator**, not competition data."
    )


def _micro_twin_landmark_name(zone_lookup: dict | None, zone_id) -> str:
    if not zone_lookup or zone_id is None:
        return "—"
    z = zone_lookup.get(str(zone_id)) or zone_lookup.get(zone_id)
    if isinstance(z, dict):
        return str(z.get("name") or z.get("landmark_name") or zone_id)
    return str(zone_id)


def _render_micro_twin_sample_card(meta: dict, sample_rate_hz: float, zone_lookup: dict | None):
    """Compact Micro-Twin metadata for judges (synthetic extension, not scored)."""
    st.markdown("**Micro-Twin sample (synthetic) — metadata**")
    sig_type = meta.get("signal_type")
    if sig_type is None or (isinstance(sig_type, float) and np.isnan(sig_type)):
        sig_disp = "noise-only sample (no structured signal type)"
    else:
        sig_disp = str(sig_type)
    rows = [
        {"Field": "label (ground truth for this synthetic row)", "Value": meta.get("label", "—")},
        {"Field": "zone_id", "Value": meta.get("zone_id", "—")},
        {"Field": "landmark_name", "Value": _micro_twin_landmark_name(zone_lookup, meta.get("zone_id"))},
        {"Field": "signal_type", "Value": sig_disp},
        {"Field": "snr_db", "Value": meta.get("snr_db", "—")},
        {"Field": "cfo_hz", "Value": meta.get("cfo_hz", "—")},
        {"Field": "num_taps", "Value": meta.get("num_taps", "—")},
        {"Field": "sample_rate_hz", "Value": sample_rate_hz},
        {"Field": "seed", "Value": meta.get("seed", "—")},
        {"Field": "file (synthetic id)", "Value": meta.get("file", "—")},
    ]
    if pd is not None:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.json({r["Field"]: r["Value"] for r in rows})
    st.success(
        "**Plain-language note:** Stretches of IQ samples near **zero do not automatically mean “empty spectrum.”** "
        "For Micro-Twin data, the **label and metadata row** tell you whether the window was generated as **noise-only** "
        "or **with structured signal** — use those fields, not the waveform shape alone."
    )


def _run_judge_submission_inference(repo_root: Path, folder_name: str | None, iq: np.ndarray | None) -> None:
    """Populate session_state with live submission evaluate() results (no tracebacks to user)."""
    st.session_state.pop("judge_live_inf_err", None)
    if (
        not folder_name
        or iq is None
        or not _SUBMISSION_ADAPTER_OK
        or _sa_run_evaluate_on_iq_array is None
    ):
        st.session_state["judge_live_pred"] = None
        st.session_state["judge_live_conf"] = None
        return
    pkg = (repo_root / "submissions" / folder_name).resolve()
    if not pkg.is_dir():
        st.session_state["judge_live_pred"] = None
        st.session_state["judge_live_inf_err"] = "Submission folder not found."
        return
    try:
        mod = _cached_submission_module(str(pkg))
        pred, info = _sa_run_evaluate_on_iq_array(mod, iq)
        st.session_state["judge_live_pred"] = int(pred)
        st.session_state["judge_live_conf"] = info.get("confidence")
        st.session_state["judge_live_inf_detail"] = {
            "trained_path_active": info.get("trained_path_active"),
            "fallback_active": info.get("fallback_active"),
            "raw_type": info.get("raw_type"),
        }
        if info.get("error"):
            st.session_state["judge_live_inf_err"] = str(info["error"])[:500]
    except Exception as e:
        st.session_state["judge_live_pred"] = None
        st.session_state["judge_live_conf"] = None
        st.session_state["judge_live_inf_err"] = f"{type(e).__name__}: {e}"[:500]


def _render_judge_gary_micro_twin_3d():
    """
    Digital Twin realism pass (Future Work only): anchor-site 3D scene, radio/user proxies,
    explicit RAN pipeline, KPI cards, 6G roadmap. Core SpectrumX detector remains separate.
    """
    if pdk is None:
        st.warning(
            "pydeck is not available in this runtime. Install `pydeck` locally to render the 3D Gary Micro-Twin building scene."
        )
        return

    # --- Non-scoring banner (do not hide judged vs future work) ---
    st.warning(
        "**Future work / research extension:** This tab is **not** the SpectrumX DAC judged detector. "
        "It is a **site-aware AI-RAN + Micro-Twin demo** for judges and 6G-style research storytelling."
    )

    st.markdown("## Gary Micro-Twin — site-aware digital twin (AI-RAN demo)")
    st.caption(
        "**Figure: Gary anchor-site Micro-Twin** — extruded footprints, hypothetical gNB / demand / interference overlays (proxies). "
        "No official competition data."
    )

    # Anchor sites: enriched for conference-demo clarity
    buildings = [
        {
            "id": "city_hall",
            "name": "Gary City Hall",
            "building_type": "Municipal civic",
            "role": "Civic command center",
            "why_matters": "Services, hearings, and emergency coordination need dependable links when residents need government most.",
            "height_m": 60,
            "footprint_approx_m2": 4200,
            "polygon": [
                [-87.3379, 41.5841],
                [-87.3374, 41.5841],
                [-87.3374, 41.5837],
                [-87.3379, 41.5837],
            ],
            "risk_bias": 0.55,
            "users": ["Residents (services)", "Civic staff", "Visitors / filers"],
            "gnb_offset_lon": 0.00022,
            "gnb_offset_lat": 0.00012,
            "demand_base_radius_m": 200,
        },
        {
            "id": "public_library",
            "name": "Gary Public Library & Cultural Center",
            "building_type": "Public library & cultural venue",
            "role": "Learning & inclusion hub",
            "why_matters": "Patrons rely on Wi‑Fi and future cellular for homework, job search, and digital literacy.",
            "height_m": 45,
            "footprint_approx_m2": 5100,
            "polygon": [
                [-87.3338, 41.5846],
                [-87.3333, 41.5846],
                [-87.3333, 41.5842],
                [-87.3338, 41.5842],
            ],
            "risk_bias": 0.40,
            "users": ["Patrons", "Study-room users", "Public-access learners"],
            "gnb_offset_lon": -0.00018,
            "gnb_offset_lat": 0.00015,
            "demand_base_radius_m": 240,
        },
        {
            "id": "west_side_leadership",
            "name": "West Side Leadership Academy",
            "building_type": "K–12 school campus",
            "role": "Education & workforce pipeline",
            "why_matters": "Students and teachers need consistent access for instruction, safety comms, and take-home equity.",
            "height_m": 35,
            "footprint_approx_m2": 3800,
            "polygon": [
                [-87.3482, 41.5852],
                [-87.3477, 41.5852],
                [-87.3477, 41.5848],
                [-87.3482, 41.5848],
            ],
            "risk_bias": 0.65,
            "users": ["Students", "Teachers", "Staff"],
            "gnb_offset_lon": 0.0002,
            "gnb_offset_lat": -0.00014,
            "demand_base_radius_m": 260,
        },
    ]

    # --- Scenario controls (above map): compact toolbar ---
    st.markdown("### Scenario & site")
    tb1, tb2, tb3, tb4, tb5 = st.columns([1, 1, 1, 1, 1.2])
    with tb1:
        b_demand = st.selectbox(
            "Demand",
            options=["Low", "Medium", "High"],
            index=1,
            key="judge_mt_demand",
            help="Traffic / throughput stress (proxy).",
        )
    with tb2:
        b_occupancy = st.selectbox(
            "Occupancy prior",
            options=["Low", "Medium", "High"],
            index=1,
            key="judge_mt_occupancy_prior",
        )
    with tb3:
        b_signal_env = st.selectbox(
            "RF environment",
            options=["Quieter / low interference", "Moderate interference", "Noisier / high interference"],
            index=1,
            key="judge_mt_signal_env",
        )
    with tb4:
        b_time = st.selectbox(
            "Time context",
            options=["School hours", "After hours", "Weekend"],
            index=0,
            key="judge_mt_time_context",
        )
    with tb5:
        b_event = st.selectbox(
            "Event mode",
            options=["Normal", "Special event (high load)"],
            index=0,
            key="judge_mt_event_mode",
        )

    site_ids = [b["id"] for b in buildings]
    site_id_default = site_ids[0]
    prev_selected = st.session_state.get("judge_mt_selected_site_id", site_id_default)
    selected_site_id = st.selectbox(
        "Focus site (updates panels below)",
        options=site_ids,
        format_func=lambda s: next(b["name"] for b in buildings if b["id"] == s),
        index=site_ids.index(prev_selected) if prev_selected in site_ids else 0,
        key="judge_mt_selected_site_id_select",
    )
    st.session_state["judge_mt_selected_site_id"] = selected_site_id
    selected_building = next((b for b in buildings if b["id"] == selected_site_id), buildings[0])

    demand_w = {"Low": 0.25, "Medium": 0.55, "High": 0.85}[b_demand]
    occ_w = {"Low": 0.25, "Medium": 0.55, "High": 0.80}[b_occupancy]
    env_w = {
        "Quieter / low interference": 0.25,
        "Moderate interference": 0.55,
        "Noisier / high interference": 0.85,
    }[b_signal_env]

    # Effective scenario weights (site + time + event)
    eff_demand = float(demand_w)
    eff_occ = float(occ_w)
    eff_env = float(env_w)
    if b_event == "Special event (high load)":
        eff_demand = min(0.95, eff_demand + 0.18)
        eff_occ = min(0.95, eff_occ + 0.08)
    if b_time == "School hours" and selected_site_id == "west_side_leadership":
        eff_occ = min(0.95, eff_occ + 0.14)
        eff_demand = min(0.95, eff_demand + 0.1)
    elif b_time == "After hours":
        eff_occ *= 0.88
    if b_time == "Weekend" and selected_site_id == "public_library":
        eff_demand = min(0.95, eff_demand + 0.1)

    for b in buildings:
        risk = 0.34 * eff_demand + 0.33 * eff_occ + 0.33 * eff_env + 0.18 * b["risk_bias"]
        b["risk_score"] = float(risk)
        if risk < 0.50:
            b["risk_label"] = "Lower coexistence stress (proxy)"
            b["fill_color"] = [46, 204, 113, 210]
        elif risk < 0.70:
            b["risk_label"] = "Moderate coexistence stress (proxy)"
            b["fill_color"] = [241, 196, 15, 210]
        else:
            b["risk_label"] = "Higher coexistence stress (proxy)"
            b["fill_color"] = [231, 76, 60, 210]
        lons = [p[0] for p in b["polygon"]]
        lats = [p[1] for p in b["polygon"]]
        b["centroid"] = [sum(lons) / len(lons), sum(lats) / len(lats)]
        # Tooltip fields (shared template with scatter overlays)
        b["label"] = b["name"]
        b["tip"] = f"{b['building_type']} · {b['role']} · {b['risk_label']}"

    # --- Central 3D canvas ---
    st.markdown("### Interactive 3D — Gary anchor sites")
    st.caption(
        "Hover pickable **buildings** (footprint + role). **Blue** = hypothetical gNB proxy; **violet** = demand hotspot; **red** = interference proxy."
    )

    demand_scale = 0.75 + 0.55 * eff_demand
    gnb_rows = []
    demand_rows = []
    for b in buildings:
        clon, clat = b["centroid"]
        gnb_rows.append(
            {
                "position": [clon + b["gnb_offset_lon"], clat + b["gnb_offset_lat"]],
                "label": f"gNB proxy · {b['name'][:28]}",
                "tip": f"Hypothetical macro/small-cell (not measured). Serves {b['building_type']}.",
            }
        )
        demand_rows.append(
            {
                "position": [clon, clat],
                "label": f"Demand zone · {b['name'][:24]}",
                "radius": int(b["demand_base_radius_m"] * demand_scale),
                "tip": f"User demand hotspot (proxy). Radius scales with scenario demand.",
            }
        )

    interference_rows = [
        {
            "position": [-87.3392, 41.5844],
            "label": "Ambient interference proxy",
            "radius": 200 + int(120 * eff_env),
            "tip": "Aggregated external RF activity (proxy). **Future:** Sionna RT / measurement trace.",
        },
        {
            "position": [-87.3355, 41.5839],
            "label": "Secondary clutter source (proxy)",
            "radius": 140 + int(80 * eff_env),
            "tip": "Low-7 GHz coexistence stressor — not a real identified emitter.",
        },
    ]

    try:
        poly_layer = pdk.Layer(
            "PolygonLayer",
            data=buildings,
            get_polygon="polygon",
            get_fill_color="fill_color",
            get_line_color=[20, 20, 20, 140],
            get_line_width=2,
            extruded=True,
            get_elevation="height_m",
            pickable=True,
            auto_highlight=True,
            opacity=0.92,
        )
        demand_layer = pdk.Layer(
            "ScatterplotLayer",
            data=demand_rows,
            get_position="position",
            get_radius="radius",
            get_fill_color=[155, 89, 182, 90],
            get_line_color=[100, 50, 130, 200],
            line_width_min_pixels=1,
            pickable=True,
        )
        if_layer = pdk.Layer(
            "ScatterplotLayer",
            data=interference_rows,
            get_position="position",
            get_radius="radius",
            get_fill_color=[231, 76, 60, 85],
            get_line_color=[160, 40, 30, 220],
            line_width_min_pixels=1,
            pickable=True,
        )
        gnb_layer = pdk.Layer(
            "ScatterplotLayer",
            data=gnb_rows,
            get_position="position",
            get_radius=95,
            get_fill_color=[52, 152, 219, 240],
            get_line_color=[30, 90, 150, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=buildings,
            get_position="centroid",
            get_text="name",
            get_color=[25, 25, 25, 255],
            get_size=15,
            pickable=False,
        )
        deck = pdk.Deck(
            layers=[poly_layer, demand_layer, if_layer, gnb_layer, text_layer],
            initial_view_state=pdk.ViewState(
                latitude=41.58425,
                longitude=-87.3398,
                zoom=12.65,
                pitch=56,
                bearing=-28,
                max_pitch=85,
            ),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={
                "html": "<b>{label}</b><br/><span style='font-size:11px'>{tip}</span>",
                "style": {"backgroundColor": "#1e1e2e", "color": "white"},
            },  # buildings + gNB/demand/interference rows all define label/tip
        )
        try:
            st.pydeck_chart(deck, use_container_width=True)
        except TypeError:
            st.pydeck_chart(deck)
    except Exception:
        st.error(
            "3D Micro-Twin scene failed to render. Install **`pydeck`** locally and reload. "
            "(Technical details are hidden for judge-facing stability.)"
        )
        return

    st.caption(
        "**Figure: community impact at City Hall / Library / West Side** — building height & tint encode **coexistence stress proxy** "
        "under your scenario (not calibrated field data)."
    )

    # --- Site identity card (selected) ---
    sc1, sc2, sc3 = st.columns([1.1, 1.1, 1.2])
    with sc1:
        st.metric("Building type", selected_building["building_type"])
        st.caption(f"**Role:** {selected_building['role']}")
    with sc2:
        st.metric("Approx. height", f"{selected_building['height_m']} m")
        st.metric("Footprint (approx.)", f"{selected_building['footprint_approx_m2']:,} m²")
    with sc3:
        st.success(f"**Why this site matters:** {selected_building['why_matters']}")
        st.caption(
            f"**Scenario:** {b_demand} demand · {b_occupancy} occ. prior · {b_signal_env} · **{b_time}** · **{b_event}**"
        )

    # --- Radio environment: visible cards (not buried in expanders) ---
    st.markdown("### Radio environment layers (proxies + roadmap tags)")
    clon, clat = selected_building["centroid"]
    gnb_lon = clon + selected_building["gnb_offset_lon"]
    gnb_lat = clat + selected_building["gnb_offset_lat"]
    los_proxy = "Partial LOS"
    if selected_building["height_m"] >= 55:
        los_proxy = "Shadowing likely (tall civic mass)"
    elif selected_site_id == "west_side_leadership":
        los_proxy = "Mixed LOS (campus + parking)"

    pen_db = 12 + 8 * eff_env + (4 if selected_building["height_m"] > 40 else 0)
    block_score = min(1.0, 0.25 + 0.45 * eff_env + 0.15 * (selected_building["height_m"] / 70.0))

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("**gNB / transmitter (proxy)**  \n`implemented proxy`")
        st.caption(f"Lat {gnb_lat:.5f}, Lon {gnb_lon:.5f}")
        st.caption("Hypothetical serving node offset from footprint centroid.")
    with r2:
        st.markdown("**User hotspots / demand**  \n`implemented proxy`")
        st.caption(f"Violet disks on map; radius ∝ demand scenario (~{int(selected_building['demand_base_radius_m'] * demand_scale)} m here).")
    with r3:
        st.markdown("**Interference / risk zones**  \n`implemented proxy`")
        st.caption("Red disks = aggregated external activity (not real identified emitters).")
    with r4:
        st.markdown("**Low-7 GHz propagation assumptions**  \n`simulated proxy`")
        st.caption(f"**LOS proxy:** {los_proxy}. **Indoor penetration (proxy):** ~{pen_db:.0f} dB equiv. **Blockage score:** {block_score:.2f} (0–1).")
        st.caption("**Future integration:** ray-traced path loss (Sionna RT) / learned channels (DeepMIMO workflow).")

    st.info(
        "**Tags:** **Implemented proxy** = shown in this UI/map now, not field-calibrated. "
        "**Future integration** = DeepMIMO / Sionna RT class tooling — **not** driving the judged detector unless you wire it in code."
    )

    # --- Users at this site ---
    st.markdown("### Users at this site")
    uc = st.columns(len(selected_building["users"]))
    for i, persona in enumerate(selected_building["users"]):
        with uc[i]:
            st.markdown(
                f"<div style='border:1px solid #dee2e6;border-radius:10px;padding:12px;background:#f8f9fa;text-align:center'>"
                f"<strong>{persona}</strong></div>",
                unsafe_allow_html=True,
            )
    st.caption(
        "**City Hall:** residents, staff, visitors · **Library:** patrons, study users, learners · **West Side:** students, teachers, staff."
    )

    # --- RAN controller: visual pipeline ---
    st.markdown("### Site-aware RAN controller loop (demo)")
    st.caption(
        "**Figure: site-aware RAN controller loop** — five-stage pipeline from sensed spectrum to KPIs (research demo only)."
    )

    pred = st.session_state.get("judge_live_pred")
    conf = st.session_state.get("judge_live_conf")
    occ_word = "occupied (1)" if pred == 1 else ("noise-only (0)" if pred == 0 else "unknown")
    belief_hi = eff_env > 0.62 or eff_demand > 0.72
    occ_belief = "High occupancy / demand pressure" if eff_occ > 0.65 else "Moderate occupancy belief"
    if eff_occ < 0.42:
        occ_belief = "Lower occupancy belief"

    candidates = [
        ("hold", "Hold transmission", "Wait / sense / avoid adding energy."),
        ("cautious", "Transmit cautiously", "Lower power or narrower beam story (proxy)."),
        ("power", "Reduce power", "Protect neighbors while maintaining a link."),
        ("channel", "Switch channel", "Avoid overlapping interference (proxy)."),
        ("prioritize", "Prioritize service at site", "Steer capacity to high-equity demand."),
    ]

    if pred is None:
        chosen_key = "hold"
        action_reason = "No live detector output — open **Core Submission** to run `evaluate()` on demo IQ."
    elif pred == 1 and belief_hi:
        chosen_key = "cautious"
        action_reason = (
            f"**High demand + stressed RF + {selected_building['name']}** → cautious transmission to protect coexistence."
        )
    elif pred == 1:
        chosen_key = "channel" if eff_env > 0.5 else "cautious"
        action_reason = "Structured energy detected; moderate stress → balance throughput and neighbor protection."
    elif pred == 0 and belief_hi:
        chosen_key = "hold"
        action_reason = "Vacancy belief but noisy environment → hold and scan before occupying spectrum."
    else:
        chosen_key = "prioritize" if eff_demand > 0.6 else "hold"
        action_reason = "Vacancy belief with capacity opportunity → prioritize equitable service at the focus site."

    pipe = st.columns(5)
    stages = [
        ("1 · Sense", f"Detector: **{occ_word}**", "Live `evaluate()` on Judge demo IQ when available."),
        ("2 · Belief", occ_belief, f"Interference belief: **{'elevated' if belief_hi else 'moderate'}**."),
        ("3 · Site ctx", selected_building["name"][:22] + "…", f"{b_time} · {b_event} · demand **{b_demand}**"),
        ("4 · Action", next(c[1] for c in candidates if c[0] == chosen_key), action_reason),
        ("5 · KPI", "See outcome row →", "Proxies only — not drive-test data."),
    ]
    for col, (title, headline, sub) in zip(pipe, stages):
        with col:
            st.markdown(
                f"<div style='border:1px solid #ced4da;border-radius:10px;padding:10px;min-height:168px;background:#ffffff'>"
                f"<div style='font-size:12px;color:#6c757d'>{title}</div>"
                f"<div style='font-weight:600;margin:6px 0'>{headline}</div>"
                f"<div style='font-size:12px;color:#495057'>{sub}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("**Candidate actions (this demo)**")
    ca = st.columns(5)
    for i, (k, title, blurb) in enumerate(candidates):
        with ca[i]:
            if k == chosen_key:
                st.markdown(f"**→ {title}**")
            else:
                st.caption(title)
            st.caption(blurb)

    kpi_cov = max(0.0, min(1.0, 0.52 + 0.28 * (1.0 - eff_env) + (0.14 if pred == 1 else -0.04)))
    kpi_coex = max(0.0, min(1.0, 0.64 - 0.22 * eff_env + (0.08 if pred == 0 else -0.04)))
    kpi_fair = max(0.0, min(1.0, 0.48 + 0.18 * eff_occ - 0.12 * (1.0 - eff_demand)))
    kpi_energy = max(0.0, min(1.0, 0.78 - 0.12 * eff_demand + (0.06 if chosen_key in ("hold", "power") else -0.04)))
    kpi_reliab = max(0.0, min(1.0, 0.5 + 0.22 * kpi_cov - 0.18 * eff_env + 0.08 * (1.0 if pred is not None else 0.0)))

    st.markdown("### Outcome & impact (proxy KPIs)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Coverage proxy", f"{kpi_cov:.2f}", help="Not measured on-air.")
    k2.metric("Coexistence score", f"{kpi_coex:.2f}", help="Neighbor-friendly spectrum use (proxy).")
    k3.metric("Community benefit", f"{kpi_fair:.2f}", help="Digital-divide / equity storytelling proxy.")
    k4.metric("Energy / efficiency", f"{kpi_energy:.2f}", help="Hold/low-power favors efficiency (proxy).")
    k5.metric("Service continuity", f"{kpi_reliab:.2f}", help="Reliability proxy vs scenario stress.")
    st.caption(
        "**Honest labeling:** All KPIs are **deterministic scenario proxies** for screenshots and PhD-style systems narrative — "
        "**not** empirical network measurements."
    )

    # --- Plain-language RF × place ---
    st.markdown("### How low-7 GHz 5G-like signals interact with this site")
    st.success(
        "**Buildings** block and scatter radio waves — taller civic structures create **shadows** and **variable indoor penetration**. "
        "**Who shows up when** (school hours, weekend library use, city events) changes **demand** and **coexistence risk**: more active users "
        "mean less margin for careless transmission. **Spectrum sensing** (like the judged detector, run separately on real data) tells a "
        "future controller **whether the band looks occupied** so it can **hold power**, **retune**, or **prioritize** service where the "
        "community needs it most."
    )

    # --- 6G research roadmap (visible, not overclaimed) ---
    st.markdown("### Next realism upgrades (6G / wireless ML research roadmap)")
    rm1, rm2, rm3 = st.columns(3)
    with rm1:
        st.markdown(
            "<div style='border-left:4px solid #3498db;padding-left:12px'>"
            "<strong>DeepMIMO path</strong><br/><span style='font-size:13px'>"
            "Site-specific channel scenarios, spatial consistency across anchor footprints, exportable CIR / features for ML.</span><br/>"
            "<em>Future integration</em> — not in this Streamlit build unless added in repo.</div>",
            unsafe_allow_html=True,
        )
    with rm2:
        st.markdown(
            "<div style='border-left:4px solid #9b59b6;padding-left:12px'>"
            "<strong>Sionna RT path</strong><br/><span style='font-size:13px'>"
            "Ray-tracing / materials / diffraction for realistic blockage and coverage maps at low-7 GHz.</span><br/>"
            "<em>Future integration</em> — complements (not replaces) the judged DAC detector.</div>",
            unsafe_allow_html=True,
        )
    with rm3:
        st.markdown(
            "<div style='border-left:4px solid #27ae60;padding-left:12px'>"
            "<strong>Beam / coverage UI</strong><br/><span style='font-size:13px'>"
            "Azimuth–elevation heatmaps, per-site SINR contours, controller replay logs for paper figures.</span><br/>"
            "<em>Planned UI hooks</em> — conference-demo targets for the next pass.</div>",
            unsafe_allow_html=True,
        )

    st.caption("**Map legend:** Green / yellow / red buildings = lower → higher **coexistence stress proxy** under the scenario.")

    st.caption(
        "**Screenshot tip:** set scenario + site, then capture **3D canvas**, **controller pipeline**, and **KPI row** as separate figures."
    )


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    _ensure_session_defaults()
    judge_mode = st.toggle(
        "Judge Mode",
        value=bool(st.session_state.get("judge_mode_toggle", False)),
        key="judge_mode_toggle",
        help="Polished, read-only judge-facing dashboard (no dev controls, no raw debug).",
    )

    if judge_mode:
        # Judge view: keep UI minimal and polished. Core metrics come from local submissions/ CSVs.
        st.header("🏆 Judge Mode")
        st.caption(
            "Core judged submission metrics are loaded from `submissions/submission_metrics.csv` (if present). "
            "Official competition IQ data is not included in this app."
        )

        _j_repo = _repo_root()
        _j_folders = _discover_submissions_safe(_j_repo)
        _j_best = _default_best_pkg(_j_repo)
        st.markdown("**Live inference package**")
        st.caption(
            "Runs `evaluate()` from `submissions/<pkg>/main.py` on **synthetic demo IQ** only (no competition data)."
        )
        _j_mode = st.radio(
            "Submission mode",
            [SUBMISSION_MODEL_FINAL, SUBMISSION_MODEL_EXPLORER],
            key="judge_sidebar_submission_mode",
            help="Final = highest-priority folder (e.g. leaderboard_v9 when present). Explorer = pick any package.",
        )
        if not _j_folders:
            st.warning("No submission packages found (need `submissions/*/main.py`).")
            st.session_state["judge_active_submission_folder"] = None
        elif _j_mode == SUBMISSION_MODEL_FINAL:
            st.session_state["judge_active_submission_folder"] = _j_best
            st.success(f"**Active package:** `{_j_best or '—'}`")
        else:
            _idx = _j_folders.index(_j_best) if _j_best in _j_folders else 0
            st.selectbox(
                "Submission package",
                _j_folders,
                index=_idx,
                key="judge_explorer_submission_folder",
            )
            st.session_state["judge_active_submission_folder"] = st.session_state.get(
                "judge_explorer_submission_folder", _j_best
            )

        uploaded_file = None
        is_int16_interleaved = False
        model_option = "Spectral Flatness"
        show_time_iq = True
        show_constellation = True
        show_psd = True
        show_spectrogram = True
        sample_rate = 1e6

        # Keep ui_mode stable in session state for existing helpers that check session_state.
        ui_mode = st.session_state.get("ui_mode", "Standard Mode")
    else:
        st.header("🧾 Report / Figure Mode")
        ui_mode = st.radio(
            "Mode",
            ["Standard Mode", "Figure Mode"],
            key="ui_mode",
            help="Figure Mode reorganizes the app into report-oriented tabs for screenshot-ready figures.",
        )
        if ui_mode == "Figure Mode":
            st.toggle(
                "Screenshot Preset",
                key="figure_screenshot_preset",
                help="Standardize spacing, emphasize titles, and use taller plots for screenshots.",
            )
            st.toggle(
                "Show captions under charts",
                key="figure_show_captions",
                help="Captions help screenshots stand alone in the final report.",
            )
            with st.expander("Report Notes (for screenshots)", expanded=True):
                st.text_input("Report section title", key="report_section_title")
                st.text_input("Suggested figure number", key="report_figure_number")
                st.text_area("Caption text", key="report_caption_text", height=80)

        st.header("⚙️ Configuration")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload .npy file",
            type=["npy"],
            help="Upload IQ data in .npy format",
            key="sidebar_file_uploader_npy",
        )

        # Format options
        is_int16_interleaved = st.checkbox(
            "int16 interleaved format",
            help="Check if data is int16 interleaved [I0, Q0, I1, Q1, ...]",
            key="sidebar_checkbox_int16_interleaved",
        )

        # Model selection (Final Submission default for judge-aligned exploration)
        _std_pkg_list = _discover_submissions_safe(_repo_root())
        _std_best = _default_best_pkg(_repo_root())
        model_option = st.selectbox(
            "Model / submission",
            [
                SUBMISSION_MODEL_FINAL,
                SUBMISSION_MODEL_EXPLORER,
                "Energy Detector",
                "Spectral Flatness",
                "PSD+LogReg",
                "Coming soon",
            ],
            help="Final Submission runs `submissions/<best>/main.py` (priority: v9 > v12 > …). Explorer lets you pick a folder.",
            key="sidebar_select_model",
        )
        if model_option == SUBMISSION_MODEL_EXPLORER:
            if not _std_pkg_list:
                st.caption("No `submissions/*/main.py` packages found.")
            else:
                _eidx = (
                    _std_pkg_list.index(_std_best)
                    if _std_best in _std_pkg_list
                    else 0
                )
                st.selectbox(
                    "Submission package",
                    _std_pkg_list,
                    index=_eidx,
                    key="std_explorer_submission_folder",
                )

        # Plot toggles
        st.header("📊 Visualizations")
        show_time_iq = st.toggle("Time/IQ", value=True, key="sidebar_toggle_time_iq")
        show_constellation = st.toggle("Constellation", value=True, key="sidebar_toggle_constellation")
        show_psd = st.toggle("PSD", value=True, key="sidebar_toggle_psd")
        show_spectrogram = st.toggle("Spectrogram", value=True, key="sidebar_toggle_spectrogram")

        # Sample rate input
        st.header("🔧 Parameters")
        default_sample_rate = st.session_state.get("demo_sample_rate", 1e6) if "demo_iq" in st.session_state else 1e6
        sample_rate = st.number_input(
            "Sample Rate (Hz)",
            min_value=1.0,
            value=float(default_sample_rate),
            step=1e3,
            format="%.0f",
            help="Sample rate of the IQ data",
            key="sidebar_sample_rate_hz",
        )

        # Micro-Twin sample picker (when available)
        if st.session_state.get("micro_twin_list"):
            mt_list = st.session_state["micro_twin_list"]
            mt_labels = [
                f"{m[1].get('zone_id', '?')} | L{m[1].get('label', '?')} | SNR:{m[1].get('snr_db', 'N/A')}"
                for m in mt_list
            ]
            st.selectbox(
                "Micro-Twin sample",
                range(len(mt_list)),
                format_func=lambda i: mt_labels[i],
                key="micro_twin_select",
            )

        if ui_mode == "Figure Mode":
            st.header("🏁 Results (headline numbers)")
            st.text_input("Baseline name", key="results_baseline_name")
            st.text_input("Baseline metric", key="results_baseline_metric")
            st.text_input("Improved name", key="results_improved_name")
            st.text_input("Improved metric", key="results_improved_metric")

# ============================================================================
# Main Content
# ============================================================================

# Determine if we have data to process
has_data = False
iq_data = None

if uploaded_file is not None:
    # Load from uploaded file
    iq_data = load_iq_data(uploaded_file, is_int16_interleaved)
    if iq_data is not None:
        has_data = True
        # Clear demo flag when file is uploaded
        if 'use_demo' in st.session_state:
            st.session_state['use_demo'] = False
elif 'demo_iq' in st.session_state and st.session_state.get('use_demo', False):
    # Use demo data
    iq_data = st.session_state.get('demo_iq')
    if iq_data is not None:
        has_data = True
        if 'demo_sample_rate' in st.session_state:
            sample_rate = st.session_state['demo_sample_rate']
elif st.session_state.get("use_micro_twin") and st.session_state.get("micro_twin_list"):
    # Use selected Micro-Twin sample
    mt_list = st.session_state["micro_twin_list"]
    sel = st.session_state.get("micro_twin_select", 0)
    if 0 <= sel < len(mt_list):
        iq_data = mt_list[sel][0]
        sample_rate = mt_list[sel][2]
        has_data = True

# Judge Mode: override input with synthetic demo for visualization only.
judge_mode_enabled = bool(st.session_state.get("judge_mode_toggle", False))
if judge_mode_enabled:
    try:
        judge_sr = float(sample_rate) if "sample_rate" in locals() else 1e6
        if "judge_demo_iq" not in st.session_state or "judge_demo_meta" not in st.session_state:
            _jiq, _jmeta = _generate_synthetic_demo_iq(sample_rate=judge_sr, duration=1.0, seed=42)
            st.session_state["judge_demo_iq"] = _jiq
            st.session_state["judge_demo_meta"] = _jmeta
            st.session_state["judge_demo_sample_rate"] = judge_sr
        iq_data = st.session_state["judge_demo_iq"]
        sample_rate = st.session_state.get("judge_demo_sample_rate", 1e6)
        has_data = True
        # Do not use Micro-Twin samples as the basis of judge-mode visuals/claims.
        st.session_state["use_micro_twin"] = False
        _run_judge_submission_inference(_repo_root(), st.session_state.get("judge_active_submission_folder"), iq_data)
    except Exception:
        # Fail gracefully: keep has_data=False so user sees an informative message.
        has_data = False

# ---------------------------------------------------------------------------
# Standard Mode (preserve existing behavior)
# ---------------------------------------------------------------------------
if judge_mode_enabled:
    # Judge-facing, read-only dashboard.
    st.header("🏆 Judge Tour")
    st.caption(
        "This view is designed for judges: polished, authoritative tables from local `submissions/` CSVs, "
        "and clearly separated Future Work / Micro-Twin (not the basis of official evaluation)."
    )

    # Load structured metrics (authoritative when present).
    repo_root = _repo_root()
    inv_rows = scan_submissions_inventory(str(repo_root))
    mdf = load_submission_metrics(str(repo_root))
    ldf = load_leaderboard_summary(str(repo_root))
    fig_yaml = load_final_report_figures_yaml(str(repo_root))
    core_row = _pick_core_submission_row(mdf, ldf, inv_rows) if (mdf is not None or ldf is not None) else _pick_core_submission_row(None, None, inv_rows)

    # Pick a submission folder name for efficiency scans (only if we have a core row).
    core_submission_name = str(core_row.get("submission")) if core_row and core_row.get("submission") is not None else None

    # Judge-tour tabs
    tabs = st.tabs(
        [
            "Problem",
            "Core Submission",
            "Results",
            "Efficiency",
            "Future Work / Micro-Twin",
            "Why It Matters for Gary",
        ]
    )

    with tabs[0]:
        st.subheader("Figure 1: Problem / Task Overview")
        if fig_yaml and fig_yaml.get("_error"):
            st.info(str(fig_yaml["_error"]))
        st.markdown(
            """
SpectrumX DAC judging focuses on: **detection accuracy**, **implementation efficiency**, **algorithmic novelty**, and **visualization quality**.

This dashboard presents:
- **Core judged submission** (official SpectrumX DAC detector metrics) from local structured files.
- **Future work** (Gary Micro-Twin / AI-RAN visualization / simulation concepts) clearly labeled as non-scoring.
            """.strip()
        )
        st.caption(
            "**Evidence map:** **Accuracy** → Results tab / CSV; **Efficiency** → Efficiency tab; **Novelty** → Core Submission + notes; **Visualization** → Problem + Future Work figures."
        )
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_1",
                "Cloud safety: official competition IQ data is not included in this app.",
            )
        )
        st.caption(
            "**Figure 1 caption:** Task overview and judging pillars — core vs future work separation."
        )

        st.subheader("Figure 2: IQ + PSD + Spectrogram (Visualization Only)")
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_2",
                "Synthetic demo IQ is shown for screenshot clarity. It does not drive judge-scoring metrics in this UI.",
            )
        )
        _render_synthetic_demo_metadata_callout(
            st.session_state.get("judge_demo_meta"),
            caption_prefix="Judge Mode:",
        )
        st.caption(
            "**Figure 2 caption:** IQ, Welch PSD, and spectrogram panels for **visualization quality** judging — "
            "synthetic demo only; no official competition IQ in-cloud."
        )
        if has_data and iq_data is not None:
            # Time-domain IQ
            c1, c2 = st.columns(2)
            with c1:
                max_points = 8000
                if len(iq_data) > max_points:
                    step = len(iq_data) // max_points
                    iq_plot = iq_data[::step]
                    t_plot = np.arange(len(iq_plot)) * step / sample_rate
                else:
                    iq_plot = iq_data
                    t_plot = np.arange(len(iq_plot)) / sample_rate
                fig_iq = make_subplots(rows=2, cols=1, subplot_titles=("I(t)", "Q(t)"), vertical_spacing=0.15)
                fig_iq.add_trace(go.Scatter(x=t_plot, y=np.real(iq_plot), mode="lines", name="I", line=dict(width=1)), row=1, col=1)
                fig_iq.add_trace(go.Scatter(x=t_plot, y=np.imag(iq_plot), mode="lines", name="Q", line=dict(width=1)), row=2, col=1)
                fig_iq.update_layout(height=360, showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig_iq, use_container_width=True)
                st.caption("Time-domain view to communicate signal bursts vs noise-only windows.")

            with c2:
                freqs, psd = compute_psd(iq_data, sample_rate)
                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(x=freqs, y=10 * np.log10(np.abs(psd) + 1e-10), mode="lines"))
                fig_psd.update_layout(height=360, xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)", margin=dict(t=40, b=10))
                st.plotly_chart(fig_psd, use_container_width=True)
                st.caption("Welch PSD communicates spectral structure relative to noise.")

            # Spectrogram (single figure)
            freqs, times, Sxx = compute_spectrogram(iq_data, sample_rate)
            fig_spec = go.Figure()
            fig_spec.add_trace(
                go.Heatmap(
                    z=10 * np.log10(np.abs(Sxx) + 1e-10),
                    x=times,
                    y=freqs,
                    colorscale="Viridis",
                )
            )
            fig_spec.update_layout(
                height=420,
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_spec, use_container_width=True)
            st.caption("Spectrogram provides a time-frequency screenshot panel.")
        else:
            st.info("No demo input available for Figure 2. Reload the app; Judge Mode should auto-generate a synthetic IQ window.")

    with tabs[1]:
        st.warning(
            "**What was judged vs future work — read first:** "
            "**(A) Core judged submission** = feature-based binary detector on **official SpectrumX labeled data** "
            "(tables/card below use **local CSVs only**). "
            "**(B) Future work** = Gary Micro-Twin, DeepMIMO / Sionna RT, AI-RAN visuals — **not** the official leaderboard evaluation basis."
        )
        st.subheader("Figure 4: Canonical Final Submission card (Core Judged — single source of truth)")
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_4",
                "Read-only, judge-facing summary populated from local structured metrics CSVs.",
            )
        )
        st.caption(
            "**Figure 4 caption:** Accuracy / rank / model family / threshold / runtime evidence for **detection** and **efficiency** judging."
        )
        if core_row is None:
            st.warning(
                "Structured metrics not found. Add `submissions/submission_metrics.csv` so the core submission card can be populated."
            )
            core_row = {}
        artifact_present = _interpret_bool(core_row.get("artifact_present"))
        if artifact_present is None:
            artifact_present = bool(core_row.get("artifact_present", False))

        trained_primary = bool(artifact_present)  # conservative: infer trained-artifact when we have an artifact.

        submitted_name = core_row.get("submission", "—")
        sub_version = core_row.get("submission_version", core_row.get("version", "—"))
        model_family = core_row.get("model_family", "—")
        leaderboard_rank = core_row.get("leaderboard_rank", "—")
        leaderboard_accuracy = core_row.get("leaderboard_accuracy", "—")
        threshold = core_row.get("threshold", "—")

        # Optional runtimes (if you extend your metrics CSV)
        runtime_val = core_row.get("runtime", core_row.get("runtime_per_sample", core_row.get("runtime_sec", None)))

        c1, c2, c3 = st.columns(3)
        c1.metric("Submission", str(submitted_name))
        c2.metric("Submission version", str(sub_version))
        c3.metric("Model family", str(model_family))

        c4, c5, c6 = st.columns(3)
        c4.metric("Leaderboard rank", "N/A" if leaderboard_rank in (None, "—") else str(leaderboard_rank))
        c5.metric("Leaderboard accuracy", "N/A" if leaderboard_accuracy in (None, "—") else str(leaderboard_accuracy))
        c6.metric("Threshold", "N/A" if threshold in (None, "—") else str(threshold))

        c7, c8, c9 = st.columns(3)
        c7.metric("Runtime (per sample)", "N/A" if runtime_val is None else str(runtime_val))
        c8.metric("Artifact used?", "Yes" if artifact_present else "No")
        c9.metric("Trained-model primary?", "Yes" if trained_primary else "No")

        st.markdown("---")
        # Inference path (code-text heuristic from submission folder)
        if core_submission_name and repo_root.is_dir():
            inf_path = _infer_inference_path(repo_root, core_submission_name)
            size_mb, size_hint = _compute_submission_artifact_footprint(repo_root, core_submission_name)
            st.metric("Inference path (heuristic)", inf_path)
            st.caption(f"Learned artifact footprint: ~{size_mb:.2f} MB ({size_hint})")
        else:
            st.metric("Inference path (heuristic)", "Unknown (missing submission folder)")

        st.caption("Metrics are loaded from local structured CSVs only. No official competition IQ data is accessed here.")

        st.markdown("---")
        st.subheader("Live submission inference (synthetic demo IQ)")
        st.caption(
            "Runs the selected **`submissions/<pkg>/main.py`** `evaluate()` on the **Judge Mode synthetic demo** window. "
            "Demonstrates **implementation** and **visualization** integration — not a leaderboard replay."
        )
        _live_folder = st.session_state.get("judge_active_submission_folder")
        if _SUBMISSION_ADAPTER_OK and _sa_submission_folder_info is not None and _live_folder:
            _pinfo = _sa_submission_folder_info(_repo_root(), str(_live_folder))
            _crow = _metrics_row_for_submission(mdf, str(_live_folder))
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Package", str(_live_folder))
            pc2.metric("Model family (CSV)", str(_crow.get("model_family", "—")) if _crow else str(_pinfo.get("model_family_guess", "—")))
            pc3.metric("Artifact present", "Yes" if _pinfo.get("artifact_present") else "No")
            st.text(f"Threshold (CSV): {_crow.get('threshold', '—') if _crow else '—'}")
            jp = st.session_state.get("judge_live_pred")
            jc = st.session_state.get("judge_live_conf")
            jd = st.session_state.get("judge_live_inf_detail") or {}
            st.metric("Prediction on demo IQ", "Occupied (1)" if jp == 1 else ("Noise-only (0)" if jp == 0 else "—"))
            st.metric("Confidence / probability", "N/A" if jc is None else str(jc))
            if jd.get("trained_path_active"):
                st.success("Heuristic: **trained-artifact path** may be active (see `main.py`).")
            if jd.get("fallback_active"):
                st.info("Heuristic: **fallback / baseline branch** may exist in `main.py`.")
            if st.session_state.get("judge_live_inf_err"):
                st.warning(f"Inference note: {st.session_state['judge_live_inf_err']}")
        elif not _live_folder:
            st.info("Select a submission package in the sidebar to enable live inference.")
        else:
            st.info("Submission adapter unavailable; live inference disabled.")

    with tabs[2]:
        st.subheader("Figure 5: Results (CV + Leaderboard Progress)")
        st.markdown(
            "Authoritative CV / leaderboard tables come from `submissions/submission_metrics.csv` when present."
        )
        st.caption(
            "**Novelty story (for judges):** feature-based, interpretable occupancy detection with a clear inference contract — "
            "see **Prediction Path** in Figure Mode and `submissions/<pkg>/main.py` for the packaged algorithm."
        )

        st.subheader("Figure 3: Feature Extraction (Visualization Only)")
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_3",
                "Feature table/chart shown for interpretability screenshots. Judge-scoring metrics in this view come only from structured CSVs.",
            )
        )
        if has_data and iq_data is not None:
            df = _features_dataframe(iq_data, sample_rate)
            if df is not None:
                if pd is not None:
                    st.dataframe(pd.DataFrame(df), use_container_width=True, hide_index=True)
                else:
                    st.caption("pandas not available in this runtime; showing feature rows as JSON.")
                    st.json(df[:50])
                # Screenshot-friendly emphasis: top magnitude features.
                try:
                    top = sorted(df, key=lambda r: abs(r["value"]), reverse=True)[:10]
                    fig_feat = go.Figure()
                    fig_feat.add_trace(
                        go.Bar(
                            x=[r["value"] for r in top],
                            y=[r["feature"] for r in top],
                            orientation="h",
                        )
                    )
                    fig_feat.update_layout(height=360, margin=dict(t=40, b=10), title="Top Feature Magnitudes (demo)")
                    st.plotly_chart(fig_feat, use_container_width=True)
                except Exception:
                    st.caption("Top-feature chart not rendered in this runtime.")
            else:
                st.info("Feature extractor not available in this runtime. (Shared `extract_features` import failed.)")
        else:
            st.info("No demo input available for Figure 3. Reload the app; Judge Mode should auto-generate synthetic IQ.")
        st.caption(
            "**Figure 3 caption:** Handcrafted feature panel for **interpretability / novelty** evidence (synthetic demo IQ)."
        )
        st.markdown("---")

        if mdf is not None:
            st.dataframe(mdf, use_container_width=True, hide_index=True)
            st.caption("If your CSV has a large number of columns, use the table view to focus on key metrics.")
        else:
            st.warning(
                "Structured metrics file not found. Add `submissions/submission_metrics.csv` for report-ready judge tables."
            )
            expected_cols = [
                "submission",
                "submission_version",
                "model_family",
                "artifact_present",
                "cv_accuracy",
                "cv_precision",
                "cv_recall",
                "cv_f1",
                "threshold",
                "leaderboard_rank",
                "leaderboard_accuracy",
                "notes",
                "change",
                "runtime",
            ]
            if pd is not None:
                st.dataframe(pd.DataFrame(columns=expected_cols), use_container_width=True, hide_index=True)
            else:
                st.info("Install `pandas` locally to view the placeholder schema table.")

        # Highlight core row if possible.
        if core_row is not None and mdf is not None and "submission" in mdf.columns and core_row.get("submission") in set(mdf["submission"].astype(str)):
            core_name = str(core_row.get("submission"))
            try:
                st.info(f"Core submission row: `{core_name}`")
            except Exception:
                pass

        st.markdown("---")
        st.subheader("Leaderboard progress (rank / version / note)")
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_5",
                "Subset for screenshots: version, rank, accuracy, and change notes when columns exist in submission_metrics.csv.",
            )
        )
        lb_slim = _leaderboard_progress_dataframe(mdf)
        if lb_slim is not None:
            st.dataframe(lb_slim, use_container_width=True, hide_index=True)
            if "leaderboard_accuracy" in lb_slim.columns:
                try:
                    id_col = "submission_version" if "submission_version" in lb_slim.columns else "submission"
                    fig_lb = go.Figure(
                        go.Bar(
                            x=lb_slim[id_col].astype(str),
                            y=lb_slim["leaderboard_accuracy"].astype(float),
                        )
                    )
                    fig_lb.update_layout(
                        title="Leaderboard accuracy (progress view)",
                        height=400,
                        margin=dict(t=50, b=10),
                    )
                    st.plotly_chart(fig_lb, use_container_width=True)
                except Exception:
                    st.caption("Could not render leaderboard bar chart (check numeric types).")
        else:
            st.info(
                "Add **leaderboard_rank** plus **submission** or **submission_version** to `submissions/submission_metrics.csv` "
                "for the progress table. Optional: **leaderboard_accuracy**, **notes** / **change**."
            )

        st.markdown("---")
        st.subheader("Submission Explorer (read-only)")
        st.caption("Inventory of `submissions/*` — code + artifacts only; no competition IQ loaded.")
        if not inv_rows:
            st.info("No submission folders found under `submissions/`.")
        elif pd is not None:
            st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)
        else:
            st.json(inv_rows[:50])

    with tabs[3]:
        st.subheader("Figure: Efficiency & Implementation Fit")
        if core_submission_name:
            runtime_val = core_row.get("runtime", core_row.get("runtime_per_sample", core_row.get("runtime_sec", None)))
            st.markdown(
                "Efficiency is communicated using *local, evidence-backed* signals: learned artifact footprint and inference-path heuristics from `submissions/<core>/main.py`."
            )
            if runtime_val is not None:
                st.metric("Runtime per sample (from CSV)", str(runtime_val))
            else:
                st.info("Runtime per sample not found in CSV. Add a `runtime` column in `submissions/submission_metrics.csv` for best judge readability.")

            size_mb, size_hint = _compute_submission_artifact_footprint(repo_root, core_submission_name)
            st.metric("Learned artifact footprint (approx)", f"~{size_mb:.2f} MB")
            if size_hint:
                st.caption(f"Artifacts: {size_hint}")

            inf_path = _infer_inference_path(repo_root, core_submission_name)
            st.metric("Inference path", inf_path)
            nlines = _user_reqs_line_count(repo_root, core_submission_name)
            if nlines is not None:
                st.metric("Dependency lines (user_reqs.txt, non-comment)", str(nlines))
            else:
                st.caption("No `user_reqs.txt` found in submission folder (optional footprint signal).")
            st.caption(
                "Complexity summary: checks whether the submission mentions feature extraction, linear models (logistic/SVM), and whether it loads persisted weights."
            )

            # Code-text heuristics (read-only inspection).
            main_txt = _safe_read_text(repo_root / "submissions" / core_submission_name / "main.py")
            nl = main_txt.lower()
            has_feats = "extract_features" in nl or "extract features" in nl
            mentions_lin = any(k in nl for k in ("logistic", "linearsvc", "svm", "sklearn"))
            c_fx, c_lin, c_notes = st.columns(3)
            c_fx.metric("Feature markers", "Yes" if has_feats else "No")
            c_lin.metric("Linear model markers", "Yes" if mentions_lin else "No")
            notes_txt = (core_row.get("notes", "") if core_row else "")
            c_notes.metric("Submission note", "Provided" if str(notes_txt).strip() else "Not provided")
            if str(notes_txt).strip():
                st.caption(str(notes_txt)[:4000])
        else:
            st.warning("Core submission folder could not be inferred. Add metrics CSV so efficiency panels can populate.")

    with tabs[4]:
        st.subheader("Future Work / Micro-Twin")
        st.caption(
            _fig_yaml_caption(
                fig_yaml,
                "figure_6",
                "Figure 6: 3D Gary Micro-Twin — synthetic / future-work visualization only (not used for official leaderboard scoring).",
            )
        )
        _render_judge_gary_micro_twin_3d()

    with tabs[5]:
        st.subheader("Novelty Story: What Was Judged vs Future Work")
        st.markdown(
            """
**Core judged submission (scoring basis):**
- Feature-based binary detector trained on official SpectrumX labeled data (metrics shown from local structured CSVs).

**Future work (non-scoring impact story):**
- Gary Micro-Twin building model and AI-RAN visualization concepts.
- DeepMIMO / Sionna RT research-grade simulation path for future scenario-aware sensing.
            """.strip()
        )
        st.caption("This panel explicitly separates scoring-algorithm evidence from future visualization/simulation work.")
        st.markdown("---")
        st.subheader("Why it Matters for Gary")
        st.markdown(
            """
The Micro-Twin future work is designed to communicate how improved sensing and coexistence can reduce digital divide gaps—especially for learning and civic services.
            """.strip()
        )

elif st.session_state.get("ui_mode") != "Figure Mode":
    if has_data and iq_data is not None:
        # Demo mode banner when using in-app demo data
        if st.session_state.get("use_demo") and "demo_iq" in st.session_state:
            st.info("📡 **Demo mode active** — using in-app generated IQ sample.")
        # Data info panel
        with st.expander("📋 File Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shape", str(iq_data.shape))
            with col2:
                st.metric("Dtype", str(iq_data.dtype))
            with col3:
                st.metric("Length", f"{len(iq_data):,} samples")
            st.text(f"Format: {'int16 interleaved' if is_int16_interleaved else 'auto-detected'}")
            st.text(f"Sample rate: {sample_rate:,.0f} Hz")
            st.text(f"Duration: {len(iq_data) / sample_rate:.4f} seconds")

        if st.session_state.get("use_micro_twin") and st.session_state.get("micro_twin_list"):
            _mtl = st.session_state["micro_twin_list"]
            _mts = st.session_state.get("micro_twin_select", 0)
            if 0 <= _mts < len(_mtl):
                _iqm, _mmeta, _msr = _mtl[_mts]
                _render_micro_twin_sample_card(
                    _mmeta,
                    float(_msr),
                    st.session_state.get("micro_twin_zone_lookup"),
                )
        if st.session_state.get("use_demo") and st.session_state.get("demo_metadata"):
            _render_synthetic_demo_metadata_callout(st.session_state.get("demo_metadata"))
        
        # Model prediction panel
        st.header("🎯 Prediction")
        
        prediction = None
        confidence = None
        confidence_is_numeric = True
        model_output = {}

        _rr_std = _repo_root()
        _active_pkg = None
        if model_option == SUBMISSION_MODEL_FINAL:
            _active_pkg = _default_best_pkg(_rr_std)
        elif model_option == SUBMISSION_MODEL_EXPLORER:
            _active_pkg = st.session_state.get("std_explorer_submission_folder") or _default_best_pkg(_rr_std)

        if model_option in (SUBMISSION_MODEL_FINAL, SUBMISSION_MODEL_EXPLORER):
            if not _SUBMISSION_ADAPTER_OK or _sa_run_evaluate_on_iq_array is None:
                st.warning("Submission adapter not available (import error).")
                prediction, confidence = 0, 0.0
                confidence_is_numeric = True
            elif not _active_pkg:
                st.warning("No submission package found under `submissions/`.")
                prediction, confidence = 0, 0.0
                confidence_is_numeric = True
            else:
                try:
                    _pkgp = (_rr_std / "submissions" / _active_pkg).resolve()
                    _mod = _cached_submission_module(str(_pkgp))
                    pred, info = _sa_run_evaluate_on_iq_array(_mod, iq_data)
                    prediction = int(pred)
                    _c = info.get("confidence")
                    if isinstance(_c, (int, float)):
                        confidence = float(_c)
                        confidence_is_numeric = True
                    else:
                        confidence = None
                        confidence_is_numeric = False
                    if _sa_submission_folder_info is not None:
                        _pi = _sa_submission_folder_info(_rr_std, str(_active_pkg))
                    else:
                        _pi = {}
                    model_output = {
                        "submission_package": _active_pkg,
                        "artifact_present": _pi.get("artifact_present"),
                        "model_family_guess": _pi.get("model_family_guess"),
                    }
                    if info.get("error"):
                        st.warning(f"Inference note: {info['error'][:400]}")
                except Exception as e:
                    st.warning(f"Submission inference failed ({type(e).__name__}). Check `main.py` and dependencies.")
                    prediction, confidence = 0, 0.0
                    confidence_is_numeric = True
        
        elif model_option == "Energy Detector":
            threshold = st.slider(
                "Energy Threshold",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Power threshold for energy detector",
                key="std_slider_energy_threshold",
            )
            prediction, confidence, power = energy_detector(iq_data, threshold)
            model_output = {"mean_power": power, "threshold": threshold}
        elif model_option == "Spectral Flatness":
            threshold = st.slider(
                "Flatness Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Spectral flatness threshold (lower = more signal-like)",
                key="std_slider_flatness_threshold",
            )
            prediction, confidence, flatness = spectral_flatness_detector(
                iq_data, sample_rate, threshold
            )
            model_output = {"flatness": flatness, "threshold": threshold}
        elif model_option == "PSD+LogReg":
            result = psd_logreg_detector(iq_data, sample_rate)
            if result is None:
                st.info("PSD+LogReg model not yet implemented. Check src/edge_ran_gary/models/ for model files.")
                prediction, confidence = 0, 0.0
            else:
                prediction, confidence = result
        elif model_option == "Coming soon":
            st.info("This model is coming soon.")
            prediction, confidence = 0, 0.0
        
        # Display prediction
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Signal" if prediction == 1 else "Noise")
        with col2:
            if confidence is None or not confidence_is_numeric:
                st.metric("Confidence / probability", "N/A")
            else:
                st.metric("Confidence / probability", f"{confidence:.3f}")
        
        if model_output:
            with st.expander("Model Details", expanded=False):
                for key, value in model_output.items():
                    if isinstance(value, float):
                        st.text(f"{key}: {value:.6f}")
                    else:
                        st.text(f"{key}: {value}")
        
        # Visualizations
        st.header("📈 Visualizations")
        
        if show_time_iq:
            st.subheader("Time Domain: I(t), Q(t), |x(t)|")
            max_points = 10000
            if len(iq_data) > max_points:
                step = len(iq_data) // max_points
                iq_plot = iq_data[::step]
                t_plot = np.arange(len(iq_plot)) * step / sample_rate
            else:
                iq_plot = iq_data
                t_plot = np.arange(len(iq_plot)) / sample_rate
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("I(t)", "Q(t)", "|x(t)|"),
                vertical_spacing=0.1
            )
            fig.add_trace(
                go.Scatter(x=t_plot, y=np.real(iq_plot), mode='lines', name='I', line=dict(width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=t_plot, y=np.imag(iq_plot), mode='lines', name='Q', line=dict(width=1)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=t_plot, y=np.abs(iq_plot), mode='lines', name='|x|', line=dict(width=1)),
                row=3, col=1
            )
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=3, col=1)
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        if show_constellation:
            st.subheader("IQ Constellation")
            max_points = 5000
            if len(iq_data) > max_points:
                step = len(iq_data) // max_points
                iq_const = iq_data[::step]
            else:
                iq_const = iq_data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.real(iq_const),
                y=np.imag(iq_const),
                mode='markers',
                marker=dict(size=2, opacity=0.5, color=np.abs(iq_const)),
                name='IQ samples'
            ))
            fig.update_layout(
                xaxis_title="I (In-phase)",
                yaxis_title="Q (Quadrature)",
                height=500,
                title="IQ Constellation Scatter Plot"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if show_psd:
            st.subheader("Power Spectral Density (Welch)")
            freqs, psd = compute_psd(iq_data, sample_rate)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs,
                y=10 * np.log10(np.abs(psd) + 1e-10),
                mode='lines',
                name='PSD'
            ))
            fig.update_layout(
                xaxis_title="Frequency (Hz)",
                yaxis_title="PSD (dB/Hz)",
                height=400,
                title="Power Spectral Density"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if show_spectrogram:
            st.subheader("Spectrogram")
            freqs, times, Sxx = compute_spectrogram(iq_data, sample_rate)
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=10 * np.log10(np.abs(Sxx) + 1e-10),
                x=times,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="Power (dB)")
            ))
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                height=500,
                title="Spectrogram"
            )
            st.plotly_chart(fig, use_container_width=True)

    # No data loaded - show upload prompt
    if not has_data:
        st.info("👈 Please upload a .npy file, use demo data, or generate Micro-Twin demo data below.")
        
        # Optional dataset folder message (if someone expects a path that doesn't exist)
        default_data_path = Path("data/synthetic/gary_micro_twin")
        if default_data_path.exists() and list(default_data_path.glob("*.npy")):
            st.caption(f"Found generated data in `{default_data_path}`. You can upload a file from there.")
        elif default_data_path.exists() is False:
            st.caption("No dataset folder found. Generate Micro-Twin demo data below to create synthetic samples.")
        
        # Demo data generation option
        st.markdown("---")
        st.subheader("🎲 Try Demo Data")
        
        def generate_demo():
            """Generate demo IQ data and set session state flags."""
            try:
                sample_rate = 1e6
                demo_iq, demo_meta = _generate_synthetic_demo_iq(
                    sample_rate=sample_rate, duration=1.0, seed=42
                )
                st.session_state["demo_iq"] = demo_iq
                st.session_state["demo_metadata"] = demo_meta
                st.session_state["demo_sample_rate"] = sample_rate
                st.session_state["use_demo"] = True
                if "uploaded_file" in st.session_state:
                    st.session_state["uploaded_file"] = None
            except Exception as e:
                st.warning(f"Demo IQ generation failed ({type(e).__name__}). Try again or reload the app.")
                st.session_state["use_demo"] = False
        
        if st.button(
            "Generate Demo IQ Sample (1 second, 1 MHz sample rate)",
            key="std_btn_generate_demo_iq",
        ):
            generate_demo()
            st.rerun()
        
        # Show message if demo data is loaded
        if st.session_state.get('use_demo', False) and 'demo_iq' in st.session_state:
            st.success("📡 Demo data loaded! Visualizations shown above.")
        
        st.markdown("---")
        st.subheader("🏛️ Generate Micro-Twin Demo Data")
        if st.button(
            "Generate Micro-Twin Demo Data (9–15 samples)",
            key="std_btn_generate_micro_twin",
        ):
            try:
                import sys
                from pathlib import Path
                repo_root = Path(__file__).resolve().parent.parent
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from src.edge_ran_gary.digital_twin.gary_micro_twin import GaryMicroTwin
                config_path = repo_root / "configs" / "gary_micro_twin.yaml"
                if not config_path.exists():
                    st.error("Config not found: configs/gary_micro_twin.yaml")
                else:
                    mt = GaryMicroTwin(config_path=str(config_path))
                    n_per_zone = 4  # 3 zones -> 12 samples
                    samples, meta_df = mt.generate_samples_per_zone(n_per_zone=n_per_zone, label_balance=0.5, seed=42)
                    sample_rate = mt.config.get("sample_rate", 1e6)
                    micro_twin_list = [(s, meta_df.iloc[i].to_dict(), float(sample_rate)) for i, s in enumerate(samples)]
                    st.session_state["micro_twin_list"] = micro_twin_list
                    st.session_state["micro_twin_zone_lookup"] = getattr(mt, "zone_metadata", {}) or {}
                    st.session_state["use_micro_twin"] = True
                    st.session_state["micro_twin_select"] = 0
                    st.session_state["use_demo"] = False
                    st.rerun()
            except Exception as e:
                st.warning(
                    f"Micro-Twin generation could not complete ({type(e).__name__}). "
                    "Check that `configs/gary_micro_twin.yaml` exists and dependencies are installed."
                )
        if st.session_state.get("use_micro_twin") and st.session_state.get("micro_twin_list"):
            st.caption("Select a sample from the sidebar to view IQ, PSD, and spectrogram. Prediction: (model not loaded) if no model is selected.")
        
        st.markdown("""
        ### Supported Formats:
        - **Complex array**: shape `(N,)` with dtype `complex64` or `complex128`
        - **Float array**: shape `(N, 2)` interpreted as `[I, Q]` pairs
        - **int16 interleaved**: shape `(N*2,)` with checkbox enabled: `[I0, Q0, I1, Q1, ...]`
        
        ### Usage:
        1. Upload your .npy file using the sidebar, or click "Generate Demo IQ Sample"
        2. Select a baseline model
        3. Adjust parameters (thresholds, sample rate)
        4. View visualizations and predictions
        """)

# ---------------------------------------------------------------------------
# Figure Mode (report-oriented tabs)
# ---------------------------------------------------------------------------
else:
    # Report notes banner (screenshot-friendly)
    st.markdown(
        f"**{st.session_state.get('report_section_title')}** · "
        f"{st.session_state.get('report_figure_number')} · "
        f"{st.session_state.get('report_caption_text')}"
    )

    _fig_repo_root = _repo_root()
    fig_yaml_fm = load_final_report_figures_yaml(str(_fig_repo_root))
    if fig_yaml_fm and fig_yaml_fm.get("_error"):
        st.info(str(fig_yaml_fm["_error"]))
    elif fig_yaml_fm is None:
        with st.expander("Optional: `docs/final_report_figures.yaml` (not found)", expanded=False):
            st.markdown(
                """
Copy `docs/final_report_figures.example.yaml` → `docs/final_report_figures.yaml` and edit captions for Figures 1–6.
Requires **PyYAML** (`pip install pyyaml`).
                """.strip()
            )

    heights = _plot_heights()
    tabs = st.tabs(
        [
            "Overview",
            "Input & Preprocessing",
            "Baseline Detectors",
            "Feature Extraction",
            "Prediction Path",
            "Results & Leaderboard",
            "Submission Explorer",
            "CV Metrics",
            "Leaderboard Progress",
            "Micro-Twin (Future Work)",
        ]
    )

    with tabs[0]:
        st.subheader("Overview")
        st.markdown(
            """
**Project summary:** EDGE-RAN Gary builds safe, explainable spectrum-occupancy detectors for 1-second IQ windows and provides a dashboard for visualization and baseline comparison.

**Competition task:** Given a **1-second IQ sample**, output a binary label:
- **1 = occupied** (signal present)
- **0 = unoccupied** (noise only)

**Cloud safety:** This public/cloud dashboard is for **demo/synthetic** IQ only. Do **not** upload the official competition IQ data to Streamlit Cloud.
            """.strip()
        )
        _caption(
            _fig_yaml_caption(
                fig_yaml_fm,
                "figure_1",
                "Judging pillars: detection accuracy, implementation efficiency, algorithmic novelty, visualization quality.",
            )
        )

    with tabs[1]:
        st.subheader("Input & Preprocessing")
        st.markdown(
            """
**Supported input formats (.npy):**
- **Complex**: shape `(N,)`, dtype `complex64/complex128`
- **Float I/Q**: shape `(N, 2)` interpreted as `[I, Q]`
- **int16 interleaved**: shape `(N*2,)` interpreted as `[I0, Q0, I1, Q1, ...]` (checkbox)
- **1D float**: treated as I-only with Q=0 (warn)

**Preprocessing summary (screenshot panel):**
1. Load `.npy` → normalize to `complex64 (N,)`
2. Compute PSD / spectrogram as needed (Welch/STFT)
3. Run baseline detector(s) and generate figures
            """.strip()
        )
        if has_data and iq_data is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Shape", str(iq_data.shape))
            c2.metric("Dtype", str(iq_data.dtype))
            c3.metric("Duration (s)", f"{len(iq_data)/sample_rate:.3f}")
            _caption("Input loaded and normalized to complex64 IQ for downstream detectors and report figures.")

            # Screenshot-friendly visuals (IQ + PSD + spectrogram)
            col_left, col_right = st.columns([1.2, 1.0])
            with col_left:
                st.markdown("**Time Domain: I(t), Q(t), |x(t)|**")
                max_points = 10000
                if len(iq_data) > max_points:
                    step = len(iq_data) // max_points
                    iq_plot = iq_data[::step]
                    t_plot = np.arange(len(iq_plot)) * step / sample_rate
                else:
                    iq_plot = iq_data
                    t_plot = np.arange(len(iq_plot)) / sample_rate
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("I(t)", "Q(t)", "|x(t)|"),
                    vertical_spacing=0.1
                )
                fig.add_trace(go.Scatter(x=t_plot, y=np.real(iq_plot), mode='lines', line=dict(width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t_plot, y=np.imag(iq_plot), mode='lines', line=dict(width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t_plot, y=np.abs(iq_plot), mode='lines', line=dict(width=1)), row=3, col=1)
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                fig.update_layout(height=heights["time"], showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("**Power Spectral Density (Welch)**")
                freqs, psd = compute_psd(iq_data, sample_rate)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=freqs, y=10 * np.log10(np.abs(psd) + 1e-10), mode='lines'))
                fig.update_layout(
                    height=heights["psd"],
                    title="Power Spectral Density",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="PSD (dB/Hz)",
                    margin=dict(t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Spectrogram**")
                freqs, times, Sxx = compute_spectrogram(iq_data, sample_rate)
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=10 * np.log10(np.abs(Sxx) + 1e-10),
                    x=times,
                    y=freqs,
                    colorscale='Viridis',
                    colorbar=dict(title="Power (dB)")
                ))
                fig.update_layout(
                    height=heights["spec"],
                    title="Spectrogram",
                    xaxis_title="Time (s)",
                    yaxis_title="Frequency (Hz)",
                    margin=dict(t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            _caption(
                _fig_yaml_caption(
                    fig_yaml_fm,
                    "figure_2",
                    "Figure 2: IQ + PSD + spectrogram (demo/synthetic in cloud; local IQ only when you run privately).",
                )
            )
            if st.session_state.get("use_micro_twin") and st.session_state.get("micro_twin_list"):
                _fmt = st.session_state["micro_twin_list"]
                _fsel = st.session_state.get("micro_twin_select", 0)
                if 0 <= _fsel < len(_fmt):
                    _, _fmeta, _fsr = _fmt[_fsel]
                    _render_micro_twin_sample_card(
                        _fmeta,
                        float(_fsr),
                        st.session_state.get("micro_twin_zone_lookup"),
                    )
            elif st.session_state.get("use_demo") and st.session_state.get("demo_metadata"):
                _render_synthetic_demo_metadata_callout(
                    st.session_state.get("demo_metadata"),
                    caption_prefix="Figure Mode:",
                )
        else:
            st.info("Load demo/synthetic IQ data (sidebar) to populate input and preprocessing figures.")

    with tabs[2]:
        st.subheader("Baseline Detectors")
        st.markdown(
            """
**Energy Detector:** compares mean signal power to a threshold.  
**Spectral Flatness Detector:** computes PSD flatness; lower flatness indicates more structured (signal-like) spectrum.
            """.strip()
        )
        if has_data and iq_data is not None:
            det_cols = st.columns(2)
            with det_cols[0]:
                st.markdown("**Energy Detector**")
                thr_e = st.slider("Energy Threshold", 0.0, 10.0, 1.0, 0.1, key="fig_thr_energy")
                pred_e, conf_e, power = energy_detector(iq_data, thr_e)
                st.metric("Prediction", "Signal" if pred_e == 1 else "Noise")
                st.metric("Confidence", f"{conf_e:.3f}")
                st.metric("Mean power", f"{power:.6f}")
                _caption("Energy detector: simple baseline; strong when SNR is high.")
            with det_cols[1]:
                st.markdown("**Spectral Flatness Detector**")
                thr_f = st.slider("Flatness Threshold", 0.0, 1.0, 0.5, 0.01, key="fig_thr_flatness")
                pred_f, conf_f, flatness = spectral_flatness_detector(iq_data, sample_rate, thr_f)
                st.metric("Prediction", "Signal" if pred_f == 1 else "Noise")
                st.metric("Confidence", f"{conf_f:.3f}")
                st.metric("Flatness", f"{flatness:.6f}")
                _caption("Spectral flatness: distinguishes noise-like vs structured spectral content.")
        else:
            st.info("Load demo/synthetic IQ data to show baseline detector outputs.")

    with tabs[3]:
        st.subheader("Feature Extraction")
        st.markdown(
            """
This tab shows the handcrafted features used (or intended) for compact, interpretable models:
- mean power, amplitude variance, crest factor, kurtosis
- spectral flatness, PSD entropy, top-k PSD peaks
- band-energy ratios, spectral centroid, spectral rolloff
- short-lag autocorrelation summary stats
            """.strip()
        )
        if has_data and iq_data is not None:
            df = _features_dataframe(iq_data, sample_rate)
            if df is None:
                st.warning("Shared feature extractor not available in this environment.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.session_state.get("figure_screenshot_preset", True):
                    # Optional bar chart for screenshot-friendly emphasis
                    top = sorted(df, key=lambda r: abs(r["value"]), reverse=True)[:12]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[r["value"] for r in top],
                        y=[r["feature"] for r in top],
                        orientation="h",
                    ))
                    fig.update_layout(
                        height=520,
                        title="Top feature magnitudes (demo/synthetic)",
                        xaxis_title="Value",
                        yaxis_title="Feature",
                        margin=dict(t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                _caption(
                    _fig_yaml_caption(
                        fig_yaml_fm,
                        "figure_3",
                        "Figure 3: handcrafted feature table / chart for interpretability (demo/synthetic IQ).",
                    )
                )
        else:
            st.info("Load demo/synthetic IQ data to populate the feature table and chart.")

    with tabs[4]:
        st.subheader("Prediction Path")
        st.markdown(
            """
**Screenshot-friendly decision flow:**

1. **Load IQ** (`.npy` → complex64 IQ)
2. **Extract features**
3. **If trained artifact exists** → compute linear score and apply threshold
4. **Else** → spectral flatness baseline
5. **Else** → energy baseline
6. **Always return** 0/1 (fail-safe default is 0 on unexpected input)
            """.strip()
        )
        _caption("This is the inference contract used by the leaderboard submission wrapper.")
        st.markdown("---")
        st.markdown("**Final submission package — live `evaluate()` (Figure Mode)**")
        st.caption(
            "Uses the same sidebar **Model / submission** choice as Standard Mode. "
            "Runs on the currently loaded IQ (demo, Micro-Twin, or upload — never competition IQ in-cloud)."
        )
        if has_data and iq_data is not None and model_option in (SUBMISSION_MODEL_FINAL, SUBMISSION_MODEL_EXPLORER):
            _ff_rr = _repo_root()
            _ff_pkg = (
                _default_best_pkg(_ff_rr)
                if model_option == SUBMISSION_MODEL_FINAL
                else st.session_state.get("std_explorer_submission_folder") or _default_best_pkg(_ff_rr)
            )
            if _SUBMISSION_ADAPTER_OK and _ff_pkg and _sa_run_evaluate_on_iq_array is not None:
                try:
                    _ff_mod = _cached_submission_module(str((_ff_rr / "submissions" / _ff_pkg).resolve()))
                    _ff_pred, _ff_info = _sa_run_evaluate_on_iq_array(_ff_mod, iq_data)
                    st.metric("Package", str(_ff_pkg))
                    st.metric("Prediction", "Occupied (1)" if int(_ff_pred) == 1 else "Noise-only (0)")
                    st.metric(
                        "Confidence / probability",
                        "N/A" if _ff_info.get("confidence") is None else str(_ff_info.get("confidence")),
                    )
                    if _ff_info.get("error"):
                        st.warning(str(_ff_info["error"])[:400])
                except Exception as e:
                    st.warning(f"Could not run submission here ({type(e).__name__}).")
            else:
                st.info("No submission package selected or adapter unavailable.")
        elif has_data and iq_data is not None:
            st.info("Choose **Final Submission (Best Known)** or **Submission Explorer** in the sidebar to show live `evaluate()` here.")

    with tabs[5]:
        st.subheader("Results & Leaderboard")
        mdf_r = load_submission_metrics(str(_repo_root()))
        if mdf_r is not None and pd is not None:
            st.success("Authoritative metrics from **`submissions/submission_metrics.csv`**.")
            st.dataframe(mdf_r, use_container_width=True, hide_index=True)
            _caption(
                _fig_yaml_caption(
                    fig_yaml_fm,
                    "figure_4",
                    "Figure 4 / 5: structured CV and leaderboard fields from local CSV (no competition IQ in-app).",
                )
            )
        else:
            st.warning(
                "Structured metrics file not found. Add **`submissions/submission_metrics.csv`** for authoritative results in this tab."
            )
            _cols = [
                "submission",
                "submission_version",
                "model_family",
                "artifact_present",
                "cv_accuracy",
                "cv_precision",
                "cv_recall",
                "cv_f1",
                "threshold",
                "leaderboard_rank",
                "leaderboard_accuracy",
                "notes",
                "change",
            ]
            if pd is not None:
                st.dataframe(pd.DataFrame(columns=_cols), use_container_width=True, hide_index=True)
        with st.expander("Optional draft comparison labels (screenshots only)", expanded=False):
            st.caption("Not authoritative; use only for side-by-side draft figures.")
            rows = [
                {"model": st.session_state.get("results_baseline_name"), "headline_metric": st.session_state.get("results_baseline_metric")},
                {"model": st.session_state.get("results_improved_name"), "headline_metric": st.session_state.get("results_improved_metric")},
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with tabs[6]:
        st.subheader("Submission Explorer")
        st.caption("Read-only inventory of `submissions/*` (code + artifacts). No competition IQ is loaded in this app.")
        inv = scan_submissions_inventory(str(_repo_root()))
        if not inv:
            st.info("No submission folders found under `submissions/`.")
        elif pd is not None:
            st.dataframe(pd.DataFrame(inv), use_container_width=True, hide_index=True)
        else:
            st.dataframe(inv, use_container_width=True, hide_index=True)

    with tabs[7]:
        st.subheader("CV Metrics")
        mdf = load_submission_metrics(str(_repo_root()))
        if mdf is not None:
            st.success("Loaded **submissions/submission_metrics.csv** (authoritative for report-ready CV / leaderboard tables).")
            st.dataframe(mdf, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "Structured metrics file not found. Add **submissions/submission_metrics.csv** for report-ready CV tables."
            )
            _cols = [
                "submission",
                "submission_version",
                "model_family",
                "artifact_present",
                "cv_accuracy",
                "cv_precision",
                "cv_recall",
                "cv_f1",
                "threshold",
                "leaderboard_rank",
                "leaderboard_accuracy",
                "notes",
                "change",
                "runtime",
            ]
            if pd is not None:
                st.dataframe(pd.DataFrame(columns=_cols), use_container_width=True, hide_index=True)
            else:
                st.code("\t".join(_cols))
                st.caption("`pip install pandas` locally to view CSV tables in-app.")

    with tabs[8]:
        st.subheader("Leaderboard Progress")
        mdf = load_submission_metrics(str(_repo_root()))
        if mdf is None:
            st.info("Populate **submission_metrics.csv** with leaderboard columns to visualize progress.")
        elif "leaderboard_rank" not in mdf.columns:
            st.info("Column **leaderboard_rank** not found in submission_metrics.csv.")
        else:
            lb_slim = _leaderboard_progress_dataframe(mdf)
            if lb_slim is not None:
                st.caption("Progress view: submission version (or folder), rank, accuracy, notes/change when present.")
                st.dataframe(lb_slim, use_container_width=True, hide_index=True)
                if "leaderboard_accuracy" in lb_slim.columns:
                    try:
                        id_col = "submission_version" if "submission_version" in lb_slim.columns else "submission"
                        fig = go.Figure(
                            go.Bar(
                                x=lb_slim[id_col].astype(str),
                                y=lb_slim["leaderboard_accuracy"].astype(float),
                            )
                        )
                        fig.update_layout(
                            title="Leaderboard accuracy (progress view)",
                            height=440,
                            xaxis_title="Submission / version",
                            yaxis_title="Leaderboard accuracy",
                            margin=dict(t=50, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.caption("Could not render leaderboard chart from CSV (check column types).")
            else:
                st.info("Could not build progress subset (need rank + submission or submission_version).")
            with st.expander("Full metrics table", expanded=False):
                dfp = mdf.copy()
                try:
                    dfp = dfp.sort_values("leaderboard_rank", na_position="last")
                except Exception:
                    pass
                st.dataframe(dfp, use_container_width=True, hide_index=True)
            _caption(
                _fig_yaml_caption(
                    fig_yaml_fm,
                    "figure_5",
                    "Figure 5: leaderboard-oriented progress (subset + optional full table).",
                )
            )

    with tabs[9]:
        st.subheader("Micro-Twin (Future Work)")
        st.markdown(
            """
**Future work / extension:** The Gary Micro-Twin generates synthetic IQ windows across a small set of equity-relevant zones.

- This tab is suitable for appendix/future-work screenshots.
- The Micro-Twin is **NOT** the basis of the core leaderboard submission.
            """.strip()
        )
        if st.session_state.get("micro_twin_list"):
            st.success("Micro-Twin samples are available. Use the sidebar to select a sample.")
        else:
            st.info("Generate Micro-Twin demo data below when no IQ is loaded (synthetic samples only).")
        st.info("For the **interactive 3D Gary building scene (Figure 6)**, enable **Judge Mode** in the sidebar.")
        _caption(
            _fig_yaml_caption(
                fig_yaml_fm,
                "figure_6",
                "Figure 6 (Judge Mode): 3D Micro-Twin buildings — future work only, not used for official scoring.",
            )
        )

    # Figure Mode: demo / micro-twin controls when no IQ loaded (unique widget keys)
    if not has_data:
        st.divider()
        st.info(
            "👈 **Figure Mode:** Upload `.npy` in the sidebar or generate **synthetic demo / Micro-Twin** data below. "
            "Do not upload official competition data to Streamlit Cloud."
        )

        def _fig_generate_demo():
            try:
                _sr = 1e6
                demo_iq, demo_meta = _generate_synthetic_demo_iq(sample_rate=_sr, duration=1.0, seed=42)
                st.session_state["demo_iq"] = demo_iq
                st.session_state["demo_metadata"] = demo_meta
                st.session_state["demo_sample_rate"] = _sr
                st.session_state["use_demo"] = True
                st.session_state["use_micro_twin"] = False
            except Exception as e:
                st.warning(f"Demo IQ generation failed ({type(e).__name__}).")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate demo IQ (1 s, 1 MHz)", key="fig_btn_generate_demo_iq"):
                _fig_generate_demo()
                st.rerun()
        with c2:
            if st.button("Generate Micro-Twin samples (9–15)", key="fig_btn_generate_micro_twin"):
                try:
                    import sys
                    repo_root = Path(__file__).resolve().parent.parent
                    if str(repo_root) not in sys.path:
                        sys.path.insert(0, str(repo_root))
                    from src.edge_ran_gary.digital_twin.gary_micro_twin import GaryMicroTwin
                    config_path = repo_root / "configs" / "gary_micro_twin.yaml"
                    if not config_path.exists():
                        st.error("Config not found: configs/gary_micro_twin.yaml")
                    else:
                        mt = GaryMicroTwin(config_path=str(config_path))
                        samples, meta_df = mt.generate_samples_per_zone(
                            n_per_zone=4, label_balance=0.5, seed=42
                        )
                        mt_sr = float(mt.config.get("sample_rate", 1e6))
                        micro_twin_list = [
                            (s, meta_df.iloc[i].to_dict(), mt_sr) for i, s in enumerate(samples)
                        ]
                        st.session_state["micro_twin_list"] = micro_twin_list
                        st.session_state["micro_twin_zone_lookup"] = getattr(mt, "zone_metadata", {}) or {}
                        st.session_state["use_micro_twin"] = True
                        st.session_state["micro_twin_select"] = 0
                        st.session_state["use_demo"] = False
                        st.rerun()
                except Exception as e:
                    st.warning(
                        f"Micro-Twin generation could not complete ({type(e).__name__}). "
                        "Verify config and dependencies."
                    )

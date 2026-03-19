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
except ImportError as e:
    st.error("Missing dependencies for visualizations.")
    st.code(str(e))
    st.info("Run: `pip install -r requirements.txt`")
    st.stop()

# Page config
st.set_page_config(
    page_title="SpX-DAC Baseline Comparison Dashboard",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 SpX-DAC Baseline Comparison Dashboard")

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


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    _ensure_session_defaults()
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
        type=['npy'],
        help="Upload IQ data in .npy format"
    )
    
    # Format options
    is_int16_interleaved = st.checkbox(
        "int16 interleaved format",
        help="Check if data is int16 interleaved [I0, Q0, I1, Q1, ...]"
    )
    
    # Model selection
    model_option = st.selectbox(
        "Model",
        ["Energy Detector", "Spectral Flatness", "PSD+LogReg", "Coming soon"],
        help="Select baseline model for prediction"
    )
    
    # Plot toggles
    st.header("📊 Visualizations")
    show_time_iq = st.toggle("Time/IQ", value=True)
    show_constellation = st.toggle("Constellation", value=True)
    show_psd = st.toggle("PSD", value=True)
    show_spectrogram = st.toggle("Spectrogram", value=True)
    
    # Sample rate input
    st.header("🔧 Parameters")
    # Get sample rate from session state if demo data exists
    default_sample_rate = st.session_state.get('demo_sample_rate', 1e6) if 'demo_iq' in st.session_state else 1e6
    sample_rate = st.number_input(
        "Sample Rate (Hz)",
        min_value=1.0,
        value=float(default_sample_rate),
        step=1e3,
        format="%.0f",
        help="Sample rate of the IQ data"
    )
    
    # Micro-Twin sample picker (when available)
    if st.session_state.get("micro_twin_list"):
        mt_list = st.session_state["micro_twin_list"]
        mt_labels = [f"{m[1].get('zone_id', '?')} | L{m[1].get('label', '?')} | SNR:{m[1].get('snr_db', 'N/A')}" for m in mt_list]
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

# ---------------------------------------------------------------------------
# Standard Mode (preserve existing behavior)
# ---------------------------------------------------------------------------
if st.session_state.get("ui_mode") != "Figure Mode":
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
        
        # Model prediction panel
        st.header("🎯 Prediction")
        
        prediction = None
        confidence = None
        model_output = {}
        
        if model_option == "Energy Detector":
            threshold = st.slider(
                "Energy Threshold",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Power threshold for energy detector"
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
                help="Spectral flatness threshold (lower = more signal-like)"
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
            st.metric("Confidence", f"{confidence:.3f}")
        
        if model_output:
            with st.expander("Model Details", expanded=False):
                for key, value in model_output.items():
                    st.text(f"{key}: {value:.6f}")
        
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
                # Generate a simple demo signal: QPSK-like burst with noise
                sample_rate = 1e6
                duration = 1.0
                n_samples = int(sample_rate * duration)
                t = np.arange(n_samples) / sample_rate
                
                # Generate QPSK-like signal (simplified)
                signal_freq = 100e3  # 100 kHz carrier
                signal_samples = int(0.3 * n_samples)  # 30% of window has signal
                signal_start = n_samples // 2 - signal_samples // 2
                
                demo_iq = np.random.normal(0, 0.1, n_samples) + 1j * np.random.normal(0, 0.1, n_samples)
                # Add structured signal in the middle
                signal_phase = 2 * np.pi * signal_freq * t[signal_start:signal_start+signal_samples]
                demo_iq[signal_start:signal_start+signal_samples] += 0.5 * np.exp(1j * signal_phase)
                demo_iq = demo_iq.astype(np.complex64)
                
                # Store in session state with all required flags
                st.session_state['demo_iq'] = demo_iq
                st.session_state['demo_sample_rate'] = sample_rate
                st.session_state['use_demo'] = True
                # Clear uploaded file to use demo path
                if 'uploaded_file' in st.session_state:
                    st.session_state['uploaded_file'] = None
            except Exception as e:
                st.error(f"Error generating demo data: {e}")
                st.session_state['use_demo'] = False
        
        if st.button("Generate Demo IQ Sample (1 second, 1 MHz sample rate)"):
            generate_demo()
            st.rerun()
        
        # Show message if demo data is loaded
        if st.session_state.get('use_demo', False) and 'demo_iq' in st.session_state:
            st.success("📡 Demo data loaded! Visualizations shown above.")
        
        st.markdown("---")
        st.subheader("🏛️ Generate Micro-Twin Demo Data")
        if st.button("Generate Micro-Twin Demo Data (9–15 samples)"):
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
                    st.session_state["use_micro_twin"] = True
                    st.session_state["micro_twin_select"] = 0
                    st.session_state["use_demo"] = False
                    st.rerun()
            except Exception as e:
                st.error(f"Micro-Twin generation failed: {e}")
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

    heights = _plot_heights()
    tabs = st.tabs(
        [
            "Overview",
            "Input & Preprocessing",
            "Baseline Detectors",
            "Feature Extraction",
            "Prediction Path",
            "Results & Leaderboard",
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
                _caption("Time-domain view used to illustrate signal bursts vs noise-only windows.")

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
                _caption("PSD highlights structured spectral content relative to noise floor.")

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
                _caption("Spectrogram provides a time-frequency view suitable for report screenshots.")
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
                _caption("Feature table for screenshot use (demo/synthetic IQ shown).")
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
                    _caption("Optional bar chart to highlight key features for report screenshots.")
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

    with tabs[5]:
        st.subheader("Results & Leaderboard")
        st.markdown("Use the sidebar inputs to stage polished headline numbers for report screenshots.")
        rows = [
            {"model": st.session_state.get("results_baseline_name"), "headline_metric": st.session_state.get("results_baseline_metric")},
            {"model": st.session_state.get("results_improved_name"), "headline_metric": st.session_state.get("results_improved_metric")},
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        _caption("Headline results table (numbers may be placeholders until final runs are complete).")

    with tabs[6]:
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
            st.info("Generate Micro-Twin demo data from Standard Mode's no-data screen (safe synthetic samples only).")
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
    
    # Model prediction panel
    st.header("🎯 Prediction")
    
    prediction = None
    confidence = None
    model_output = {}
    
    if model_option == "Energy Detector":
        threshold = st.slider(
            "Energy Threshold",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Power threshold for energy detector"
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
            help="Spectral flatness threshold (lower = more signal-like)"
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
        st.metric("Confidence", f"{confidence:.3f}")
    
    if model_output:
        with st.expander("Model Details", expanded=False):
            for key, value in model_output.items():
                st.text(f"{key}: {value:.6f}")
    
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
            # Generate a simple demo signal: QPSK-like burst with noise
            sample_rate = 1e6
            duration = 1.0
            n_samples = int(sample_rate * duration)
            t = np.arange(n_samples) / sample_rate
            
            # Generate QPSK-like signal (simplified)
            signal_freq = 100e3  # 100 kHz carrier
            signal_samples = int(0.3 * n_samples)  # 30% of window has signal
            signal_start = n_samples // 2 - signal_samples // 2
            
            demo_iq = np.random.normal(0, 0.1, n_samples) + 1j * np.random.normal(0, 0.1, n_samples)
            # Add structured signal in the middle
            signal_phase = 2 * np.pi * signal_freq * t[signal_start:signal_start+signal_samples]
            demo_iq[signal_start:signal_start+signal_samples] += 0.5 * np.exp(1j * signal_phase)
            demo_iq = demo_iq.astype(np.complex64)
            
            # Store in session state with all required flags
            st.session_state['demo_iq'] = demo_iq
            st.session_state['demo_sample_rate'] = sample_rate
            st.session_state['use_demo'] = True
            # Clear uploaded file to use demo path
            if 'uploaded_file' in st.session_state:
                st.session_state['uploaded_file'] = None
        except Exception as e:
            st.error(f"Error generating demo data: {e}")
            st.session_state['use_demo'] = False
    
    if st.button("Generate Demo IQ Sample (1 second, 1 MHz sample rate)"):
        generate_demo()
        st.rerun()
    
    # Show message if demo data is loaded
    if st.session_state.get('use_demo', False) and 'demo_iq' in st.session_state:
        st.success("📡 Demo data loaded! Visualizations shown above.")
    
    st.markdown("---")
    st.subheader("🏛️ Generate Micro-Twin Demo Data")
    if st.button("Generate Micro-Twin Demo Data (9–15 samples)"):
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
                st.session_state["use_micro_twin"] = True
                st.session_state["micro_twin_select"] = 0
                st.session_state["use_demo"] = False
                st.rerun()
        except Exception as e:
            st.error(f"Micro-Twin generation failed: {e}")
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

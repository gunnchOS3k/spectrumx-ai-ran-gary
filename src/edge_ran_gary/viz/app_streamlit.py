"""
Streamlit visualization integration.

This module provides a clean interface for integrating detection pipeline
results into the Streamlit dashboard. Noah will maintain this module.
"""

from typing import Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.edge_ran_gary.detection.predict import DetectionPipeline


class StreamlitVisualizer:
    """
    Visualization helper for Streamlit dashboard.
    
    Provides methods to create interactive Plotly charts for:
    - Time domain plots (I, Q, magnitude)
    - IQ constellation scatter
    - Power spectral density
    - Spectrogram
    - Prediction results with confidence
    """
    
    def __init__(self, detection_pipeline: Optional[DetectionPipeline] = None):
        """
        Initialize visualizer.
        
        Args:
            detection_pipeline: Optional detection pipeline for predictions
        """
        self.detection_pipeline = detection_pipeline
    
    def create_time_plots(
        self, 
        iq: np.ndarray, 
        sample_rate: float = 1e6,
        max_points: int = 10000
    ) -> go.Figure:
        """
        Create time domain plots (I, Q, magnitude).
        
        Args:
            iq: Complex IQ samples (N,)
            sample_rate: Sample rate in Hz
            max_points: Maximum points to plot (for performance)
            
        Returns:
            Plotly figure with subplots for I(t), Q(t), |x(t)|
        """
        # Sample for performance
        if len(iq) > max_points:
            step = len(iq) // max_points
            iq_plot = iq[::step]
            t_plot = np.arange(len(iq_plot)) * step / sample_rate
        else:
            iq_plot = iq
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
        
        return fig
    
    def create_constellation_plot(
        self, 
        iq: np.ndarray, 
        max_points: int = 5000
    ) -> go.Figure:
        """
        Create IQ constellation scatter plot.
        
        Args:
            iq: Complex IQ samples (N,)
            max_points: Maximum points to plot
            
        Returns:
            Plotly figure with constellation scatter
        """
        if len(iq) > max_points:
            step = len(iq) // max_points
            iq_plot = iq[::step]
        else:
            iq_plot = iq
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.real(iq_plot),
            y=np.imag(iq_plot),
            mode='markers',
            marker=dict(size=2, opacity=0.5, color=np.abs(iq_plot)),
            name='IQ samples'
        ))
        fig.update_layout(
            xaxis_title="I (In-phase)",
            yaxis_title="Q (Quadrature)",
            height=500,
            title="IQ Constellation Scatter Plot"
        )
        return fig
    
    def create_prediction_display(
        self, 
        prediction: int, 
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create prediction display data for Streamlit.
        
        Args:
            prediction: Binary prediction (0 or 1)
            confidence: Confidence score [0, 1]
            metadata: Optional metadata from detection pipeline
            
        Returns:
            Dictionary with display information
        """
        return {
            "prediction": "Signal" if prediction == 1 else "Noise",
            "confidence": confidence,
            "metadata": metadata or {}
        }

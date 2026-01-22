"""
Gary Spectrum Digital Twin - Synthetic IQ Data Generator

This module provides a lightweight prototype for generating synthetic 1-second IQ windows
with metadata. It supports:
- Zone-based modeling (equity-focused neighborhoods)
- Reproducible signal generation (QPSK/OFDM-like with impairments)
- Dataset building for ML pipeline testing
- Integration with detector training and AI-RAN controller

Usage:
    from edge_ran_gary.digital_twin import build_synth_dataset, generate_iq_window
    
    # Generate single window
    iq_data = generate_iq_window(seed=123, label=1, config_path="configs/digital_twin_gary.yaml")
    
    # Build full dataset
    build_synth_dataset(
        output_dir="data/synth_gary_twin",
        n_samples=2000,
        seed=123,
        config_path="configs/digital_twin_gary.yaml"
    )
"""

from .generator import generate_iq_window, SignalGenerator
from .zones import ZoneModel, load_zones_from_config
from .dataset_builder import build_synth_dataset

__all__ = [
    "generate_iq_window",
    "SignalGenerator",
    "ZoneModel",
    "load_zones_from_config",
    "build_synth_dataset",
]

"""
Detection module for spectrum occupancy detection from 1-second IQ samples.

This module implements the competition core: binary classification of IQ samples
as occupied (signal present) or unoccupied (noise only).

Components:
- features: Feature extraction from IQ samples
- baselines: Classical detection baselines (energy, spectral flatness)
- ssl: Self-supervised learning encoders
- anomaly: Anomaly detection models for unsupervised scenarios
- calibrate: Confidence calibration
- predict: Inference pipeline
"""

from src.edge_ran_gary.detection.features import FeatureExtractor
from src.edge_ran_gary.detection.baselines import (
    EnergyDetector,
    PSDLogRegDetector,
    SpectralFlatnessDetector,
)
from src.edge_ran_gary.detection.ssl import EncoderSSL
from src.edge_ran_gary.detection.anomaly import AnomalyModel
from src.edge_ran_gary.detection.calibrate import Calibrator
from src.edge_ran_gary.detection.predict import DetectionPipeline

__all__ = [
    "FeatureExtractor",
    "EnergyDetector",
    "PSDLogRegDetector",
    "SpectralFlatnessDetector",
    "EncoderSSL",
    "AnomalyModel",
    "Calibrator",
    "DetectionPipeline",
]

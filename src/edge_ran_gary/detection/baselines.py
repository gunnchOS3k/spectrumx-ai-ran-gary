"""
Baseline detection methods for spectrum occupancy.

Classical signal processing approaches that serve as baselines
and are also used in the Streamlit dashboard.
"""

from typing import Tuple
import numpy as np
from scipy import signal

from src.edge_ran_gary.config import BaselineConfig


class EnergyDetector:
    """
    Simple energy detector baseline.
    
    Compares mean power to a threshold to determine occupancy.
    """
    
    def __init__(self, threshold: float = 1.0):
        """
        Initialize energy detector.
        
        Args:
            threshold: Power threshold for detection
        """
        self.threshold = threshold
    
    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        """
        Detect occupancy using energy detector.
        
        Args:
            iq: Complex IQ samples (N,)
            
        Returns:
            Tuple of (prediction, confidence, mean_power):
            - prediction: 0 (unoccupied) or 1 (occupied)
            - confidence: Confidence score [0, 1]
            - mean_power: Mean power of the signal
        """
        power = np.mean(np.abs(iq) ** 2)
        prediction = 1 if power > self.threshold else 0
        distance = abs(power - self.threshold) / (self.threshold + 1e-10)
        confidence = min(1.0, distance)
        return prediction, confidence, power


class SpectralFlatnessDetector:
    """
    Spectral flatness detector baseline.
    
    Uses spectral flatness (geometric mean / arithmetic mean of PSD)
    to distinguish structured signals from noise.
    Lower flatness indicates more signal-like behavior.
    """
    
    def __init__(self, threshold: float = 0.5, sample_rate: float = 1e6):
        """
        Initialize spectral flatness detector.
        
        Args:
            threshold: Flatness threshold (lower = more signal-like)
            sample_rate: Sample rate in Hz for PSD computation
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
    
    def detect(self, iq: np.ndarray) -> Tuple[int, float, float]:
        """
        Detect occupancy using spectral flatness.
        
        Args:
            iq: Complex IQ samples (N,)
            
        Returns:
            Tuple of (prediction, confidence, flatness):
            - prediction: 0 (unoccupied) or 1 (occupied)
            - confidence: Confidence score [0, 1]
            - flatness: Spectral flatness value
        """
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(
            iq, 
            fs=self.sample_rate, 
            nperseg=1024,
            return_onesided=False,
            scaling='density'
        )
        
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
        prediction = 1 if flatness < self.threshold else 0
        distance = abs(flatness - self.threshold) / (self.threshold + 1e-10)
        confidence = min(1.0, distance)
        
        return prediction, confidence, flatness

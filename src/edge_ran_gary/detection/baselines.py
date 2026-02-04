"""
Baseline detection methods for spectrum occupancy.

Classical signal processing approaches that serve as baselines
and are also used in the Streamlit dashboard.
"""

from typing import Iterable, Tuple
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression

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

    def predict(self, iq: np.ndarray) -> Tuple[int, float, float]:
        """
        Alias for detect() to match guide examples.
        """
        return self.detect(iq)


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

    def predict(self, iq: np.ndarray, sample_rate: float | None = None) -> Tuple[int, float, float]:
        """
        Alias for detect() to match guide examples.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate
        return self.detect(iq)


class PSDLogRegDetector:
    """
    PSD + Logistic Regression baseline.

    Extracts simple PSD statistics and trains a logistic regression classifier.
    """

    def __init__(self, nperseg: int = 1024):
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False
        self.nperseg = nperseg

    def extract_psd_features(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Extract PSD features for logistic regression."""
        _, psd = signal.welch(
            iq_data,
            fs=sample_rate,
            nperseg=self.nperseg,
            return_onesided=False,
            scaling="density",
        )
        psd_mag = np.abs(psd)
        if psd_mag.size == 0:
            return np.zeros(6, dtype=float)
        return np.array(
            [
                np.mean(psd_mag),
                np.std(psd_mag),
                np.max(psd_mag),
                np.min(psd_mag),
                np.percentile(psd_mag, 25),
                np.percentile(psd_mag, 75),
            ],
            dtype=float,
        )

    def fit(self, X_iq: Iterable[np.ndarray], y: Iterable[int], sample_rate: float) -> None:
        """Fit on labeled data."""
        X_features = np.array(
            [self.extract_psd_features(iq, sample_rate) for iq in X_iq],
            dtype=float,
        )
        self.model.fit(X_features, np.array(list(y)))
        self.fitted = True

    def predict(self, iq_data: np.ndarray, sample_rate: float) -> Tuple[int, float]:
        """Predict on a single sample."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        features = self.extract_psd_features(iq_data, sample_rate)
        prob = float(self.model.predict_proba([features])[0, 1])
        pred = 1 if prob >= 0.5 else 0
        return pred, prob

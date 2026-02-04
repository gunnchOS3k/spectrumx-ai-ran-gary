"""
Signal generator for synthetic IQ data.

Generates reproducible 1-second IQ windows with:
- Noise-only (label=0)
- Structured transmission (label=1) with QPSK/OFDM-like waveforms
- Impairments: CFO, timing offset, multipath, AWGN
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal
from pathlib import Path
import yaml

from .zones import ZoneModel, Zone


class SignalGenerator:
    """
    Generator for synthetic IQ signals with impairments.
    
    Produces reproducible 1-second IQ windows compatible with the detector.
    """
    
    def __init__(
        self,
        sample_rate: float = 1e6,
        duration: float = 1.0,
        zone_model: Optional[ZoneModel] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize signal generator.
        
        Args:
            sample_rate: Sample rate in Hz (default: 1 MHz)
            duration: Window duration in seconds (default: 1.0)
            zone_model: ZoneModel instance (optional)
            config_path: Path to config file (optional, loads zone_model if provided)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
        if config_path:
            self.zone_model = ZoneModel.from_config(config_path)
        elif zone_model:
            self.zone_model = zone_model
        else:
            # Default: single zone
            from .zones import Zone
            default_zone = Zone(
                zone_id="default",
                weight=1.0,
                occupancy_prior=0.5,
                noise_floor_prior=-90.0,
                snr_range=(0, 20),
                cfo_range=(-1000, 1000),
                multipath_taps_range=(1, 5)
            )
            from .zones import ZoneModel
            self.zone_model = ZoneModel([default_zone])
    
    def generate_noise_only(self, seed: int, zone: Optional[Zone] = None) -> np.ndarray:
        """
        Generate noise-only IQ sample (label=0).
        
        Args:
            seed: Random seed
            zone: Zone to use (required - caller must provide selected zone)
            
        Returns:
            complex64 array of shape (N,)
        """
        rng = np.random.default_rng(seed)
        
        if zone is None:
            # Fallback: sample a zone if not provided
            zone = self.zone_model.sample_zone(rng)
        
        # Generate AWGN with noise floor
        # noise_floor_prior is in dB (relative to 1.0 linear power)
        noise_power_db = zone.noise_floor_prior
        noise_power_linear = 10 ** (noise_power_db / 10)  # Convert dB to linear
        
        # Generate complex AWGN
        noise = np.sqrt(noise_power_linear / 2) * (
            rng.normal(0, 1, self.n_samples) + 1j * rng.normal(0, 1, self.n_samples)
        )
        
        return noise.astype(np.complex64)
    
    def generate_structured_signal(
        self,
        seed: int,
        zone: Optional[Zone] = None,
        signal_type: str = "qpsk"
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate structured transmission (label=1).
        
        Args:
            seed: Random seed
            zone: Zone to use (optional, samples if None)
            signal_type: "qpsk" or "ofdm"
            
        Returns:
            Tuple of (iq_data, metadata):
            - iq_data: complex64 array
            - metadata: dict with snr_db, cfo_hz, num_taps, etc.
        """
        rng = np.random.default_rng(seed)
        
        if zone is None:
            zone = self.zone_model.sample_zone(rng)
        
        # Sample impairments from zone ranges
        snr_db = rng.uniform(*zone.snr_range)
        cfo_hz = rng.uniform(*zone.cfo_range)
        num_taps = rng.integers(*zone.multipath_taps_range)
        
        # Generate base signal (qpsk_burst / ofdm_burst use same as qpsk / ofdm)
        st = signal_type.replace("_burst", "") if isinstance(signal_type, str) else signal_type
        if st == "qpsk":
            iq_signal = self._generate_qpsk_signal(rng, zone)
        elif st == "ofdm":
            iq_signal = self._generate_ofdm_signal(rng, zone)
        elif st == "chirp":
            iq_signal = self._generate_chirp_signal(rng, zone)
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")
        
        # Apply impairments
        iq_signal = self._apply_cfo(iq_signal, cfo_hz, rng)
        iq_signal = self._apply_multipath(iq_signal, num_taps, rng)
        iq_signal = self._apply_awgn(iq_signal, snr_db, zone.noise_floor_prior, rng)
        
        metadata = {
            "snr_db": snr_db,
            "cfo_hz": cfo_hz,
            "num_taps": num_taps,
            "signal_type": signal_type,
            "zone_id": zone.zone_id
        }
        
        return iq_signal.astype(np.complex64), metadata
    
    def _generate_qpsk_signal(self, rng: np.random.Generator, zone: Zone) -> np.ndarray:
        """Generate QPSK-like signal with RRC pulse shaping."""
        # QPSK symbol rate (use 10% of bandwidth)
        symbol_rate = self.sample_rate * 0.1
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        n_symbols = self.n_samples // samples_per_symbol
        
        # Generate QPSK symbols (proper complex constellation: ±1±j)
        # Generate I and Q bits separately
        bi = rng.integers(0, 2, n_symbols)  # I bit: 0 or 1
        bq = rng.integers(0, 2, n_symbols)  # Q bit: 0 or 1
        I = 2 * bi - 1  # Convert to ±1
        Q = 2 * bq - 1  # Convert to ±1
        symbols = (I + 1j * Q) / np.sqrt(2)  # Normalized complex QPSK
        symbols = symbols.astype(np.complex64)
        
        # Upsample
        upsampled = np.zeros(n_symbols * samples_per_symbol, dtype=np.complex64)
        upsampled[::samples_per_symbol] = symbols
        
        # RRC pulse shaping
        rrc_taps = signal.firrcos(
            numtaps=8 * samples_per_symbol + 1,
            cutoff=symbol_rate / 2,
            width=symbol_rate * 0.2,
            fs=self.sample_rate
        )
        shaped = signal.lfilter(rrc_taps, 1.0, upsampled)
        
        # Pad or truncate to exact length
        if len(shaped) > self.n_samples:
            shaped = shaped[:self.n_samples]
        elif len(shaped) < self.n_samples:
            shaped = np.pad(shaped, (0, self.n_samples - len(shaped)), mode='constant')
        
        return shaped
    
    def _generate_ofdm_signal(self, rng: np.random.Generator, zone: Zone) -> np.ndarray:
        """Generate OFDM-like signal."""
        # OFDM parameters
        n_subcarriers = 64
        n_symbols = self.n_samples // (n_subcarriers + 16)  # +16 for CP
        
        # Generate random QPSK symbols per subcarrier (proper complex constellation)
        ofdm_symbols = []
        for _ in range(n_symbols):
            # Generate proper complex QPSK symbols
            bi = rng.integers(0, 2, n_subcarriers)  # I bit: 0 or 1
            bq = rng.integers(0, 2, n_subcarriers)  # Q bit: 0 or 1
            I = 2 * bi - 1  # Convert to ±1
            Q = 2 * bq - 1  # Convert to ±1
            symbols = (I + 1j * Q) / np.sqrt(2)  # Normalized complex QPSK
            # IFFT
            time_domain = np.fft.ifft(symbols, n=n_subcarriers)
            # Add cyclic prefix
            cp_length = 16
            ofdm_symbol = np.concatenate([time_domain[-cp_length:], time_domain])
            ofdm_symbols.append(ofdm_symbol)
        
        # Concatenate
        signal_full = np.concatenate(ofdm_symbols)
        
        # Pad or truncate
        if len(signal_full) > self.n_samples:
            signal_full = signal_full[:self.n_samples]
        elif len(signal_full) < self.n_samples:
            signal_full = np.pad(signal_full, (0, self.n_samples - len(signal_full)), mode='constant')
        
        return signal_full

    def _generate_chirp_signal(self, rng: np.random.Generator, zone: Zone) -> np.ndarray:
        """Generate linear chirp (complex-valued) over the window."""
        t = np.arange(self.n_samples, dtype=np.float64) / self.sample_rate
        f0 = rng.uniform(-0.2, 0.2) * self.sample_rate  # start freq
        f1 = f0 + rng.uniform(-0.1, 0.1) * self.sample_rate  # end freq
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / self.duration)
        chirp = np.exp(1j * phase).astype(np.complex64)
        # Normalize power
        chirp = chirp / np.sqrt(np.mean(np.abs(chirp) ** 2))
        return chirp

    def _apply_cfo(self, iq: np.ndarray, cfo_hz: float, rng: np.random.Generator) -> np.ndarray:
        """Apply carrier frequency offset."""
        t = np.arange(len(iq)) / self.sample_rate
        cfo_phase = 2 * np.pi * cfo_hz * t
        return iq * np.exp(1j * cfo_phase)
    
    def _apply_multipath(self, iq: np.ndarray, num_taps: int, rng: np.random.Generator) -> np.ndarray:
        """Apply multipath channel (simple FIR filter)."""
        # Generate random channel taps
        taps = rng.normal(0, 1, num_taps) + 1j * rng.normal(0, 1, num_taps)
        taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))  # Normalize power
        
        # Apply FIR filter
        filtered = signal.lfilter(taps, 1.0, iq)
        return filtered
    
    def _apply_awgn(
        self,
        iq: np.ndarray,
        snr_db: float,
        noise_floor_db: Optional[float],
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Apply AWGN to achieve target SNR.
        
        Args:
            iq: Input signal (complex array)
            snr_db: Target signal-to-noise ratio in dB
            noise_floor_db: Minimum noise floor in dB (relative to 1.0 linear power).
                           If provided, noise power will be at least this level.
            rng: Random number generator
            
        Returns:
            Signal with AWGN added
        """
        # Compute signal power (linear)
        signal_power = np.mean(np.abs(iq) ** 2)
        
        # Compute target noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
        
        # Apply noise floor if provided
        if noise_floor_db is not None:
            # Convert noise_floor_db to linear power (relative to 1.0)
            noise_floor_linear = 10 ** (noise_floor_db / 10)
            # Use maximum of target and floor
            final_noise_power = max(target_noise_power, noise_floor_linear)
        else:
            final_noise_power = target_noise_power
        
        # Generate complex AWGN
        # Each component (I and Q) has variance = final_noise_power/2
        noise = np.sqrt(final_noise_power / 2) * (
            rng.normal(0, 1, len(iq)) + 1j * rng.normal(0, 1, len(iq))
        )
        
        return iq + noise


def generate_iq_window(
    seed: int,
    label: int,
    config_path: Optional[str] = None,
    sample_rate: float = 1e6,
    duration: float = 1.0,
    zone_id: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate a single IQ window (convenience function).
    
    Args:
        seed: Random seed for reproducibility
        label: 0 (noise only) or 1 (structured signal)
        config_path: Path to config file (optional)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        zone_id: Specific zone to use (optional). If provided and invalid, raises ValueError.
        
    Returns:
        Tuple of (iq_data, metadata):
        - iq_data: complex64 array
        - metadata: dict with label, seed, zone_id, snr_db, etc.
        
    Raises:
        ValueError: If zone_id is provided but not found in zone model.
    """
    generator = SignalGenerator(
        sample_rate=sample_rate,
        duration=duration,
        config_path=config_path
    )
    
    rng = np.random.default_rng(seed)
    selected_zone = None
    
    if zone_id:
        # Validate zone_id
        zone = generator.zone_model.get_zone(zone_id)
        if zone is None:
            valid_zones = ", ".join(generator.zone_model.zone_ids)
            raise ValueError(
                f"Invalid zone_id '{zone_id}'. Valid zone IDs are: {valid_zones}"
            )
        selected_zone = zone
    else:
        # Sample a zone randomly
        selected_zone = generator.zone_model.sample_zone(rng)
    
    if label == 0:
        iq_data = generator.generate_noise_only(seed, selected_zone)
        metadata = {
            "label": 0,
            "seed": seed,
            "zone_id": selected_zone.zone_id  # Always use actual selected zone
        }
    else:
        iq_data, signal_metadata = generator.generate_structured_signal(seed, selected_zone)
        metadata = {
            "label": 1,
            "seed": seed,
            **signal_metadata
        }
    
    return iq_data, metadata

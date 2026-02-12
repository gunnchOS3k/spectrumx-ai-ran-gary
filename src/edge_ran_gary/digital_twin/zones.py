"""
Zone model for Gary Spectrum Digital Twin.

Represents neighborhoods/zones with equity-focused weights and occupancy priors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import yaml
from pathlib import Path


@dataclass
class Zone:
    """Represents a zone (neighborhood) with occupancy characteristics."""
    zone_id: str
    weight: float  # Equity weight (higher = more emphasis)
    occupancy_prior: float  # Prior probability of occupancy [0, 1]
    noise_floor_prior: float  # Noise floor in dB (relative to 1.0 linear)
    snr_range: tuple  # (min, max) SNR in dB
    cfo_range: tuple  # (min, max) carrier frequency offset in Hz
    multipath_taps_range: tuple  # (min, max) number of multipath taps
    # Optional landmark fields (for contract metadata)
    landmark_name: Optional[str] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_m: Optional[float] = None
    waveform_mix: Optional[Dict[str, float]] = None  # e.g. {"qpsk": 0.5, "ofdm": 0.5}


class ZoneModel:
    """
    Zone model for generating synthetic data with equity considerations.
    
    Zones represent abstract neighborhoods (not real PII) with different
    occupancy characteristics and equity weights.
    """
    
    def __init__(self, zones: List[Zone]):
        """
        Initialize zone model.
        
        Args:
            zones: List of Zone objects
        """
        self.zones = {zone.zone_id: zone for zone in zones}
        self.zone_ids = list(self.zones.keys())
        
        # Normalize weights for sampling
        weights = np.array([zone.weight for zone in zones])
        self.zone_probs = weights / weights.sum()
    
    def sample_zone(self, rng: np.random.Generator) -> Zone:
        """
        Sample a zone according to equity weights.
        
        Args:
            rng: Random number generator
            
        Returns:
            Sampled Zone
        """
        zone_id = rng.choice(self.zone_ids, p=self.zone_probs)
        return self.zones[zone_id]
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID."""
        return self.zones.get(zone_id)
    
    @classmethod
    def from_config(cls, config_path: str) -> "ZoneModel":
        """
        Load zone model from YAML config.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            ZoneModel instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        zones_config = config.get('zones', {})
        zones = []
        
        for zone_id, zone_data in zones_config.items():
            snr = zone_data.get('snr_range') or zone_data.get('snr_db_range', [0, 20])
            zone = Zone(
                zone_id=zone_id,
                weight=zone_data.get('weight', 1.0),
                occupancy_prior=zone_data.get('occupancy_prior', 0.5),
                noise_floor_prior=zone_data.get('noise_floor_prior', -90.0),
                snr_range=tuple(snr),
                cfo_range=tuple(zone_data.get('cfo_range', [-1000, 1000])),
                multipath_taps_range=tuple(zone_data.get('multipath_taps_range', [1, 5])),
                landmark_name=zone_data.get('landmark_name') or zone_data.get('name'),
                center_lat=zone_data.get('center_lat') or zone_data.get('lat'),
                center_lon=zone_data.get('center_lon') or zone_data.get('lon'),
                radius_m=zone_data.get('radius_m'),
                waveform_mix=zone_data.get('waveform_mix') or zone_data.get('signal_type_prior'),
            )
            zones.append(zone)
        
        return cls(zones)


def load_zones_from_config(config_path: str) -> ZoneModel:
    """Convenience function to load zones from config."""
    return ZoneModel.from_config(config_path)

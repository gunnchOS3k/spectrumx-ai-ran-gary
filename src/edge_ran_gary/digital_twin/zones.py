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
    noise_floor_prior: float  # Noise floor in dBm
    snr_range: tuple  # (min, max) SNR in dB
    cfo_range: tuple  # (min, max) carrier frequency offset in Hz
    multipath_taps_range: tuple  # (min, max) number of multipath taps


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
            zone = Zone(
                zone_id=zone_id,
                weight=zone_data.get('weight', 1.0),
                occupancy_prior=zone_data.get('occupancy_prior', 0.5),
                noise_floor_prior=zone_data.get('noise_floor_prior', -90.0),
                snr_range=tuple(zone_data.get('snr_range', [0, 20])),
                cfo_range=tuple(zone_data.get('cfo_range', [-1000, 1000])),
                multipath_taps_range=tuple(zone_data.get('multipath_taps_range', [1, 5]))
            )
            zones.append(zone)
        
        return cls(zones)


def load_zones_from_config(config_path: str) -> ZoneModel:
    """Convenience function to load zones from config."""
    return ZoneModel.from_config(config_path)

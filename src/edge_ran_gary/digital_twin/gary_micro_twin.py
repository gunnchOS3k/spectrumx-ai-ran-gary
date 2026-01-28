"""
Gary Micro-Twin: Focused 3-zone synthetic data generator.

Generates zone-aware synthetic IQ data for controlled ML testing.
Zones: City Hall, West Side Leadership Academy, Gary Public Library.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml

from .generator import SignalGenerator, generate_iq_window
from .zones import ZoneModel, Zone


class GaryMicroTwin:
    """
    Gary Micro-Twin: 3-zone synthetic data generator.
    
    Produces zone-aware synthetic IQ and metadata for controlled ML testing.
    Does NOT replace SpectrumX competition data - it's a controlled extension.
    """
    
    def __init__(self, config_path: str = "configs/gary_micro_twin.yaml"):
        """
        Initialize Gary Micro-Twin.
        
        Args:
            config_path: Path to micro-twin YAML config
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load zone model
        self.zone_model = ZoneModel.from_config(str(self.config_path))
        
        # Initialize generator
        self.generator = SignalGenerator(
            sample_rate=self.config.get('sample_rate', 1e6),
            duration=self.config.get('duration', 1.0),
            zone_model=self.zone_model
        )
        
        # Zone metadata (lat/lon/radius from config)
        self.zone_metadata = {}
        for zone_id, zone_data in self.config.get('zones', {}).items():
            self.zone_metadata[zone_id] = {
                'name': zone_data.get('name', zone_id),
                'lat': zone_data.get('lat'),
                'lon': zone_data.get('lon'),
                'radius_m': zone_data.get('radius_m', 500)
            }
    
    def generate_samples_per_zone(
        self,
        n_per_zone: int = 100,
        label_balance: float = 0.5,
        seed: int = 42
    ) -> Tuple[List[np.ndarray], pd.DataFrame]:
        """
        Generate samples for each zone.
        
        Args:
            n_per_zone: Number of samples per zone
            label_balance: Fraction of samples with label=1
            seed: Random seed
            
        Returns:
            Tuple of (iq_samples_list, metadata_df):
            - iq_samples_list: List of complex64 arrays
            - metadata_df: DataFrame with columns: file, label, zone_id, snr_db, cfo_hz, num_taps, seed
        """
        rng = np.random.default_rng(seed)
        samples = []
        metadata_rows = []
        
        for zone_id in self.zone_model.zone_ids:
            zone = self.zone_model.get_zone(zone_id)
            n_label1 = int(n_per_zone * label_balance)
            n_label0 = n_per_zone - n_label1
            
            # Generate label=0 samples
            for i in range(n_label0):
                sample_seed = seed + len(samples)
                iq_data = self.generator.generate_noise_only(sample_seed, zone)
                samples.append(iq_data)
                
                metadata_rows.append({
                    'file': f'{zone_id}_sample_{len(samples)-1:04d}.npy',
                    'label': 0,
                    'zone_id': zone_id,
                    'snr_db': np.nan,
                    'cfo_hz': np.nan,
                    'num_taps': np.nan,
                    'seed': sample_seed
                })
            
            # Generate label=1 samples
            for i in range(n_label1):
                sample_seed = seed + len(samples)
                # Sample signal type from zone's prior (if available)
                zone_config = self.config['zones'].get(zone_id, {})
                signal_type_prior = zone_config.get('signal_type_prior', {'qpsk': 0.5, 'ofdm': 0.5})
                signal_types = list(signal_type_prior.keys())
                probs = list(signal_type_prior.values())
                signal_type = rng.choice(signal_types, p=np.array(probs) / sum(probs))
                
                iq_data, signal_metadata = self.generator.generate_structured_signal(
                    sample_seed, zone, signal_type
                )
                samples.append(iq_data)
                
                metadata_rows.append({
                    'file': f'{zone_id}_sample_{len(samples)-1:04d}.npy',
                    'label': 1,
                    'zone_id': zone_id,
                    'snr_db': signal_metadata['snr_db'],
                    'cfo_hz': signal_metadata['cfo_hz'],
                    'num_taps': signal_metadata['num_taps'],
                    'seed': sample_seed,
                    'signal_type': signal_type
                })
        
        metadata_df = pd.DataFrame(metadata_rows)
        return samples, metadata_df
    
    def save_dataset(
        self,
        output_dir: str,
        samples: List[np.ndarray],
        metadata_df: pd.DataFrame
    ):
        """
        Save dataset to disk (.npy files + metadata.csv/json).
        
        Args:
            output_dir: Output directory
            samples: List of IQ samples
            metadata_df: Metadata DataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save .npy files
        for idx, (sample, row) in enumerate(zip(samples, metadata_df.itertuples())):
            filepath = output_path / row.file
            np.save(filepath, sample)
        
        # Save metadata CSV
        csv_path = output_path / "metadata.csv"
        metadata_df.to_csv(csv_path, index=False)
        
        # Save metadata JSON (with zone info)
        json_data = {
            'config_path': str(self.config_path),
            'n_samples': len(samples),
            'zones': self.zone_metadata,
            'samples': metadata_df.to_dict('records')
        }
        json_path = output_path / "metadata.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"✅ Dataset saved to {output_path}")
        print(f"  Samples: {len(samples)}")
        print(f"  Metadata CSV: {csv_path}")
        print(f"  Metadata JSON: {json_path}")
    
    def generate_coverage_plot_data(self) -> Dict:
        """
        Generate simple "coverage proxy" data for plotting zones.
        
        Returns:
            Dict with zone_id -> {lat, lon, radius_m, name, sample_count}
        """
        coverage_data = {}
        for zone_id in self.zone_model.zone_ids:
            zone_info = self.zone_metadata.get(zone_id, {})
            coverage_data[zone_id] = {
                'lat': zone_info.get('lat'),
                'lon': zone_info.get('lon'),
                'radius_m': zone_info.get('radius_m', 500),
                'name': zone_info.get('name', zone_id),
                'weight': self.zone_model.get_zone(zone_id).weight
            }
        return coverage_data


def generate_micro_twin_dataset(
    output_dir: str = "data/gary_micro_twin",
    n_per_zone: int = 100,
    label_balance: float = 0.5,
    seed: int = 42,
    config_path: str = "configs/gary_micro_twin.yaml"
):
    """
    Convenience function to generate Gary Micro-Twin dataset.
    
    Args:
        output_dir: Output directory
        n_per_zone: Samples per zone
        label_balance: Fraction with label=1
        seed: Random seed
        config_path: Path to config YAML
    """
    micro_twin = GaryMicroTwin(config_path=config_path)
    samples, metadata_df = micro_twin.generate_samples_per_zone(
        n_per_zone=n_per_zone,
        label_balance=label_balance,
        seed=seed
    )
    micro_twin.save_dataset(output_dir, samples, metadata_df)
    return micro_twin, samples, metadata_df

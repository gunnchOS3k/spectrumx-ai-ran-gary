"""
Dataset builder for synthetic IQ data.

Generates .npy files and metadata.csv for ML pipeline training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import argparse
from tqdm import tqdm

from .generator import SignalGenerator
from .zones import ZoneModel


def build_synth_dataset(
    output_dir: str,
    n_samples: int = 2000,
    seed: int = 123,
    config_path: Optional[str] = None,
    sample_rate: float = 1e6,
    duration: float = 1.0,
    label_balance: float = 0.5
):
    """
    Build synthetic IQ dataset with metadata.
    
    Args:
        output_dir: Output directory for .npy files and metadata.csv
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        config_path: Path to zone config YAML (optional)
        sample_rate: Sample rate in Hz
        duration: Duration per window in seconds
        label_balance: Fraction of samples with label=1 (default: 0.5)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SignalGenerator(
        sample_rate=sample_rate,
        duration=duration,
        config_path=config_path
    )
    
    # Set up RNG
    rng = np.random.default_rng(seed)
    
    # Determine label distribution
    n_label1 = int(n_samples * label_balance)
    n_label0 = n_samples - n_label1
    
    metadata_rows = []
    
    # Generate samples
    print(f"Generating {n_samples} synthetic IQ windows...")
    print(f"  Label 0 (noise): {n_label0}")
    print(f"  Label 1 (signal): {n_label1}")
    
    sample_idx = 0
    
    # Generate label=0 samples
    for i in tqdm(range(n_label0), desc="Noise samples"):
        sample_seed = seed + sample_idx
        zone = generator.zone_model.sample_zone(rng)
        
        iq_data = generator.generate_noise_only(sample_seed, zone)
        
        # Save .npy file
        filename = f"sample_{sample_idx:06d}.npy"
        filepath = output_path / filename
        np.save(filepath, iq_data)
        
        metadata_rows.append({
            "file": filename,
            "label": 0,
            "zone_id": zone.zone_id,
            "snr_db": np.nan,
            "cfo_hz": np.nan,
            "num_taps": np.nan,
            "seed": sample_seed
        })
        
        sample_idx += 1
    
    # Generate label=1 samples
    for i in tqdm(range(n_label1), desc="Signal samples"):
        sample_seed = seed + sample_idx
        zone = generator.zone_model.sample_zone(rng)
        
        signal_type = rng.choice(["qpsk", "ofdm"])
        iq_data, signal_metadata = generator.generate_structured_signal(
            sample_seed, zone, signal_type
        )
        
        # Save .npy file
        filename = f"sample_{sample_idx:06d}.npy"
        filepath = output_path / filename
        np.save(filepath, iq_data)
        
        metadata_rows.append({
            "file": filename,
            "label": 1,
            "zone_id": zone.zone_id,
            "snr_db": signal_metadata["snr_db"],
            "cfo_hz": signal_metadata["cfo_hz"],
            "num_taps": signal_metadata["num_taps"],
            "seed": sample_seed
        })
        
        sample_idx += 1
    
    # Save metadata CSV
    df = pd.DataFrame(metadata_rows)
    metadata_path = output_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"\n✅ Dataset generated:")
    print(f"  Output directory: {output_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Label distribution: {n_label0} noise, {n_label1} signal")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic IQ dataset for Gary Spectrum Digital Twin"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/synth_gary_twin",
        help="Output directory"
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=2000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/digital_twin_gary.yaml",
        help="Path to zone config YAML"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate in Hz"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration per window in seconds"
    )
    parser.add_argument(
        "--label-balance",
        type=float,
        default=0.5,
        help="Fraction of samples with label=1"
    )
    
    args = parser.parse_args()
    
    build_synth_dataset(
        output_dir=args.out,
        n_samples=args.n_samples,
        seed=args.seed,
        config_path=args.config,
        sample_rate=args.sample_rate,
        duration=args.duration,
        label_balance=args.label_balance
    )


if __name__ == "__main__":
    main()

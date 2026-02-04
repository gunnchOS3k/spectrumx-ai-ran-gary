"""
Run baseline detectors on competition_dataset samples.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.edge_ran_gary.detection.baselines import EnergyDetector, SpectralFlatnessDetector


def load_iq(path: Path) -> np.ndarray:
    data = np.load(path)
    if np.iscomplexobj(data):
        return data.astype(np.complex64)
    if data.ndim == 2 and data.shape[1] == 2:
        return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    if data.dtype == np.int16 and data.ndim == 1 and data.size % 2 == 0:
        i = data[0::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)
    raise ValueError(f"Unsupported IQ format: shape={data.shape}, dtype={data.dtype}")


def collect_npy_files(dataset_root: Path) -> list[Path]:
    files_root = dataset_root / "files"
    if not files_root.exists():
        return []

    user_dirs = [p for p in files_root.iterdir() if p.is_dir()]
    training_variants = ["trainingData"]
    labeled_variants = ["VLA_brutal"]

    found: list[Path] = []
    for user_dir in user_dirs:
        for tname in training_variants:
            tdir = user_dir / tname
            if tdir.exists():
                found.extend(sorted(tdir.glob("*.npy")))
        for lname in labeled_variants:
            ldir = user_dir / lname
            if ldir.exists():
                found.extend(sorted(ldir.glob("*.npy")))

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline detectors on dataset samples.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\anany\NSF project\spectrumx-ai-ran-gary\competition_dataset"),
        help="Path to competition_dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random samples to evaluate",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=1.0,
        help="Energy detector threshold",
    )
    parser.add_argument(
        "--flatness-threshold",
        type=float,
        default=0.5,
        help="Spectral flatness detector threshold",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate for PSD computation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_predictions.json"),
        help="Path to save predictions JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling files",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")

    files = collect_npy_files(dataset_root)
    print(f"Dataset root: {dataset_root}")
    print(f"Found .npy files: {len(files)}")
    if not files:
        print("No .npy files found. Check folder names under competition_dataset/files.")
        return

    energy = EnergyDetector(threshold=args.energy_threshold)
    flatness = SpectralFlatnessDetector(
        threshold=args.flatness_threshold,
        sample_rate=args.sample_rate,
    )

    sample_count = min(args.num_samples, len(files))
    rng = random.Random(args.seed)
    sample_paths = rng.sample(files, sample_count)
    results = []
    for path in sample_paths:
        iq = load_iq(path)
        pred_e, conf_e, power = energy.predict(iq)
        pred_f, conf_f, flat = flatness.predict(iq, sample_rate=args.sample_rate)

        print("\nFile:", path)
        print(f"Energy -> pred={pred_e} conf={conf_e:.3f} power={power:.6f}")
        print(f"Flatness -> pred={pred_f} conf={conf_f:.3f} flatness={flat:.6f}")

        results.append(
            {
                "file": str(path),
                "energy": {
                    "prediction": int(pred_e),
                    "confidence": float(conf_e),
                    "power": float(power),
                    "threshold": float(args.energy_threshold),
                },
                "flatness": {
                    "prediction": int(pred_f),
                    "confidence": float(conf_f),
                    "flatness": float(flat),
                    "threshold": float(args.flatness_threshold),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(dataset_root),
        "num_samples": int(sample_count),
        "seed": int(args.seed),
        "sample_rate": float(args.sample_rate),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()

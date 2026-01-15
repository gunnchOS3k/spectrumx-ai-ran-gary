from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class SpectrumXDatasetConfig:
    dataset_root: Path = Path("competition_dataset")
    sds_host: str = "sds.crc.nd.edu"


@dataclass
class PreprocessingConfig:
    normalization_method: str = "zero_mean_unit_var"  # or "power_normalize"
    window_type: Optional[str] = None  # "hann", "hamming", etc.
    window_length: Optional[int] = None


@dataclass
class FeatureExtractorConfig:
    feature_types: List[str] = field(default_factory=lambda: ["time", "freq", "statistical"])
    n_fft: int = 1024
    overlap: float = 0.5


@dataclass
class ModelConfig:
    encoder_arch: str = "transformer"  # or "cnn", "lstm"
    classifier_arch: str = "mlp"
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    hidden_dim: int = 128


@dataclass
class AnomalyModelConfig:
    model_type: str = "isolation_forest"  # or "autoencoder", "one_class_svm"
    contamination_rate: float = 0.1


@dataclass
class CalibrationConfig:
    method: str = "temperature_scaling"  # or "platt_scaling", "isotonic"
    validation_split: float = 0.2


@dataclass
class EnsembleConfig:
    fusion_method: str = "weighted_voting"  # or "stacking", "bayesian"
    model_weights: Optional[List[float]] = None


@dataclass
class EvaluationConfig:
    metrics_list: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc_roc"])
    output_dir: Path = Path("results")


@dataclass
class BaselineConfig:
    energy_threshold: float = 1.0
    spectral_flatness_threshold: float = 0.5
    sample_rate: float = 1e6


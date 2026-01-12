from dataclasses import dataclass
from pathlib import Path

@dataclass
class SpectrumXDatasetConfig:
    dataset_root: Path = Path("competition_dataset")
    sds_host: str = "sds.crc.nd.edu"


# @dataclass
# class TrainingConfig:
#     seed: int = 42
#     batch_size: int = 256
#     epochs: int = 50
#     learning_rate: float = 1e-3


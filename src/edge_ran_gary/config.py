from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3


from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 100
    LR: float = 2e-4
    LAMBDA_L1: float = 100.0
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = field(default_factory=lambda: min(4, __import__('os').cpu_count() or 2))
    LOG_DIR: str = "logs"
    CHECKPOINT_DIR: str = "checkpoints"
    DATA_DIR: str = "data"


cfg = Config()

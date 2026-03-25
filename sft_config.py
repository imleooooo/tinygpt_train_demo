from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Checkpoints
    pretrain_checkpoint: str = "checkpoint.pt"
    sft_checkpoint: str = "sft_checkpoint.pt"

    # Data
    sft_data_file: str = "data/sft_data.json"

    # Training — lower LR than pre-training is standard SFT practice
    batch_size: int = 8
    max_iters: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 20
    sample_interval: int = 100
    sample_length: int = 150
    temperature: float = 0.7
    top_k: int = 40

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip <= 0.0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {self.log_interval}")
        if self.sample_interval < 1:
            raise ValueError(f"sample_interval must be >= 1, got {self.sample_interval}")
        if self.sample_length < 1:
            raise ValueError(f"sample_length must be >= 1, got {self.sample_length}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

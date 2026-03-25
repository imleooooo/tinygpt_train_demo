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

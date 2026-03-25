from dataclasses import dataclass, field
import os


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_file: str = "data/tiny_shakespeare.txt"
    tokenizer_file: str = "data/tokenizer.json"
    checkpoint_file: str = "checkpoint.pt"

    # Model architecture
    block_size: int = 128       # context window (sequence length)
    n_embd: int = 128           # embedding dimension
    n_head: int = 4             # number of attention heads
    n_layer: int = 4            # number of transformer blocks
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    max_iters: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_split: float = 0.1      # fraction of data for validation

    # Logging
    log_interval: int = 100     # print loss every N steps
    sample_interval: int = 500  # generate sample text every N steps
    sample_length: int = 200    # characters to generate in sample

    # Generation defaults
    temperature: float = 0.8
    top_k: int = 40

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.n_embd < 1:
            raise ValueError(f"n_embd must be >= 1, got {self.n_embd}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
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
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
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

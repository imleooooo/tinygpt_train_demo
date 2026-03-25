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

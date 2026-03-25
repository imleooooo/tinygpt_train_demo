from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # Checkpoints
    sft_checkpoint:  str   = "sft_checkpoint.pt"
    grpo_checkpoint: str   = "grpo_checkpoint.pt"

    # Data
    prompts_file:    str   = "data/grpo_prompts.json"

    # GRPO hyperparameters
    G:               int   = 4      # group size: responses sampled per prompt
    max_gen_tokens:  int   = 80     # rollout length (characters)
    beta:            float = 0.04   # KL penalty coefficient

    # Training
    batch_size:      int   = 4      # prompts per update step
    max_iters:       int   = 100
    learning_rate:   float = 5e-5
    weight_decay:    float = 0.01
    grad_clip:       float = 1.0

    # Logging / sampling
    log_interval:    int   = 10
    sample_interval: int   = 50
    sample_length:   int   = 120
    temperature:     float = 0.9    # higher temp → more diverse rollouts
    top_k:           int   = 40

    # Reproducibility
    seed:            int   = 42

    def __post_init__(self):
        if self.G < 1:
            raise ValueError(f"G (group size) must be >= 1, got {self.G}")
        if self.max_gen_tokens < 1:
            raise ValueError(f"max_gen_tokens must be >= 1, got {self.max_gen_tokens}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
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
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")

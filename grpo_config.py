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

"""Entry point for GRPO training on top of a fine-tuned TinyGPT."""

import json

import torch

from grpo_config import GRPOConfig
from src.generate import load_model
from src.grpo_trainer import GRPOTrainer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    cfg = GRPOConfig()
    device = get_device()
    print(f"Using device: {device}")

    # Peek at the SFT checkpoint's pretrain config to get model architecture
    # and dropout — same pattern as sft.py
    print(f"Loading SFT checkpoint: {cfg.sft_checkpoint}")
    pretrain_cfg = torch.load(
        cfg.sft_checkpoint, map_location="cpu", weights_only=False
    )["config"]

    # Policy model: loaded with training dropout, starts in train() mode
    policy, tokenizer, _ = load_model(
        cfg.sft_checkpoint, device, dropout=pretrain_cfg.dropout
    )

    # Reference model: frozen SFT policy — no gradients, eval mode
    reference, _, _ = load_model(cfg.sft_checkpoint, device, dropout=0.0)
    for param in reference.parameters():
        param.requires_grad_(False)

    print(f"Model parameters: {policy.num_parameters():,}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Group size G={cfg.G}, batch={cfg.batch_size}, beta={cfg.beta}")

    # Load prompts
    with open(cfg.prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Prompts: {len(prompts)}")

    # Train
    trainer = GRPOTrainer(policy, reference, prompts, tokenizer, cfg, pretrain_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()

"""Entry point for text generation from a trained TinyGPT checkpoint."""

import argparse

import torch

from config import TrainConfig
from src.generate import generate_text, load_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Generate text with TinyGPT")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Seed text")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    args = parser.parse_args()

    device = get_device()
    cfg = TrainConfig()
    checkpoint = args.checkpoint or cfg.checkpoint_file

    print(f"Loading checkpoint: {checkpoint}")
    model, tokenizer, ckpt_cfg = load_model(checkpoint, device)
    print(f"Model parameters: {model.num_parameters():,}")

    temperature = args.temperature if args.temperature is not None else ckpt_cfg.temperature
    top_k = args.top_k if args.top_k is not None else ckpt_cfg.top_k

    print(f"\nPrompt: {repr(args.prompt)}")
    print(f"Generating {args.max_new_tokens} tokens (temperature={temperature}, top_k={top_k})...\n")
    print("-" * 60)

    output = generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens, temperature, top_k, device
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()

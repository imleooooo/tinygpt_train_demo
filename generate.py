"""Entry point for text generation from a trained TinyGPT checkpoint."""

import argparse
import logging

import torch

from config import TrainConfig
from src.generate import generate_text, load_model

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with TinyGPT")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Seed text")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = get_device()
    cfg = TrainConfig()
    checkpoint = args.checkpoint or cfg.checkpoint_file

    logger.info("Loading checkpoint: %s", checkpoint)
    model, tokenizer, ckpt_cfg = load_model(checkpoint, device)
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    temperature = args.temperature if args.temperature is not None else ckpt_cfg.temperature
    top_k = args.top_k if args.top_k is not None else ckpt_cfg.top_k

    logger.info("Prompt: %r | tokens: %d | temperature: %s | top_k: %s",
                args.prompt, args.max_new_tokens, temperature, top_k)
    print("-" * 60)

    output = generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens, temperature, top_k, device
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()

"""Entry point for GRPO training on top of a fine-tuned TinyGPT."""

import argparse
import json
import logging
import random

import torch

from grpo_config import GRPOConfig
from src.generate import load_model
from src.grpo_trainer import GRPOTrainer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Align TinyGPT with GRPO")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    if not logging.root.handlers:
        # CLI invocation: no handlers yet, safe to configure root logging.
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        # Embedded use: attach our own handler directly to our loggers and stop
        # propagation so --log-level is honoured regardless of what level filters
        # the host's root handlers enforce.
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        _handler._tinygpt_owned = True  # sentinel so we can find it on re-entry
        for _name in ("src", __name__):
            _lg = logging.getLogger(_name)
            # Remove only a handler we previously installed; leave host handlers intact.
            for _h in _lg.handlers[:]:
                if getattr(_h, "_tinygpt_owned", False):
                    _lg.removeHandler(_h)
            _lg.setLevel(args.log_level)
            _lg.addHandler(_handler)
            _lg.propagate = False

    cfg = GRPOConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    device = get_device()
    logger.info("Using device: %s", device)

    # Peek at the SFT checkpoint's pretrain config to get model architecture
    # and dropout — same pattern as sft.py
    logger.info("Loading SFT checkpoint: %s", cfg.sft_checkpoint)
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

    logger.info("Model parameters: %s", f"{policy.num_parameters():,}")
    logger.info("Vocab size: %d", tokenizer.vocab_size)
    logger.info("Group size G=%d, batch=%d, beta=%s", cfg.G, cfg.batch_size, cfg.beta)

    # Load prompts
    with open(cfg.prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    logger.info("Prompts: %d", len(prompts))

    # Train
    trainer = GRPOTrainer(policy, reference, prompts, tokenizer, cfg, pretrain_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()

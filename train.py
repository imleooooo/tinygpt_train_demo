"""Entry point for training TinyGPT on Tiny Shakespeare."""

import argparse
import logging
import random

import torch

from config import TrainConfig
from src.dataset import TextDataset, download_shakespeare
from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from src.trainer import Trainer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pre-train TinyGPT on Tiny Shakespeare")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig()

    if not logging.root.handlers:
        # CLI invocation: no handlers yet, safe to configure root logging.
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
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

    device = get_device()
    logger.info("Using device: %s", device)

    # 1. Download data
    download_shakespeare(cfg.data_url, cfg.data_file)

    # 2. Load text and build tokenizer
    with open(cfg.data_file, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info("Dataset size: %s characters", f"{len(text):,}")

    tokenizer = CharTokenizer().build(text)
    tokenizer.save(cfg.tokenizer_file)
    logger.info("Vocab size: %d characters", tokenizer.vocab_size)

    # 3. Encode entire corpus into a flat token tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 4. Train / validation split
    split = int(len(data) * (1 - cfg.val_split))
    train_data = TextDataset(data[:split], cfg.block_size)
    val_data = TextDataset(data[split:], cfg.block_size)
    logger.info("Train tokens: %s  |  Val tokens: %s", f"{split:,}", f"{len(data) - split:,}")

    # 5. Build model
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    ).to(device)
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    # 6. Train
    trainer = Trainer(model, train_data, val_data, tokenizer, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()

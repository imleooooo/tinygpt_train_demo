"""Entry point for SFT (Supervised Fine-Tuning) on top of a pretrained TinyGPT."""

import argparse
import logging

import torch

from config import TrainConfig
from sft_config import SFTConfig
from src.generate import load_model, _load_ckpt
from src.sft_dataset import SFTDataset
from src.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TinyGPT with SFT")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    cfg = SFTConfig()

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

    device = get_device()
    logger.info("Using device: %s", device)

    # 1. Load pretrained model + tokenizer.
    # We need pretrain_cfg.dropout before calling load_model() so we can
    # instantiate nn.Dropout with the correct p. Peek at the config cheaply,
    # then do the full load once with the right dropout value.
    logger.info("Loading pretrained checkpoint: %s", cfg.pretrain_checkpoint)
    pretrain_cfg = TrainConfig(**_load_ckpt(cfg.pretrain_checkpoint, map_location="cpu")["config"])
    model, tokenizer, pretrain_cfg = load_model(
        cfg.pretrain_checkpoint, device, dropout=pretrain_cfg.dropout
    )
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    # 2. Build SFT dataset
    dataset = SFTDataset(cfg.sft_data_file, tokenizer, pretrain_cfg.block_size)
    logger.info("SFT examples: %d", len(dataset))

    # 3. Show how many response tokens vs total tokens (to verify masking)
    total_tokens = sum(m.sum().item() for _, _, m in dataset)
    total_positions = len(dataset) * pretrain_cfg.block_size
    logger.info(
        "Response tokens (loss active): %d / %d positions",
        int(total_tokens), total_positions,
    )

    # 4. Fine-tune
    trainer = SFTTrainer(model, dataset, tokenizer, cfg, pretrain_cfg, device)
    trainer.train()


if __name__ == "__main__":
    import random
    _cfg = SFTConfig()
    random.seed(_cfg.seed)
    torch.manual_seed(_cfg.seed)
    torch.backends.cudnn.deterministic = True
    main()

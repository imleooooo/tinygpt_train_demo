"""Entry point for SFT (Supervised Fine-Tuning) on top of a pretrained TinyGPT."""

import torch

from sft_config import SFTConfig
from src.generate import load_model
from src.sft_dataset import SFTDataset
from src.sft_trainer import SFTTrainer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    cfg = SFTConfig()
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load pretrained model + tokenizer.
    # We need pretrain_cfg.dropout before calling load_model() so we can
    # instantiate nn.Dropout with the correct p. Peek at the config cheaply,
    # then do the full load once with the right dropout value.
    print(f"Loading pretrained checkpoint: {cfg.pretrain_checkpoint}")
    pretrain_cfg = torch.load(
        cfg.pretrain_checkpoint, map_location="cpu", weights_only=False
    )["config"]
    model, tokenizer, pretrain_cfg = load_model(
        cfg.pretrain_checkpoint, device, dropout=pretrain_cfg.dropout
    )
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 2. Build SFT dataset
    dataset = SFTDataset(cfg.sft_data_file, tokenizer, pretrain_cfg.block_size)
    print(f"SFT examples: {len(dataset)}")

    # 3. Show how many response tokens vs total tokens (to verify masking)
    total_tokens = sum(m.sum().item() for _, _, m in dataset)
    total_positions = len(dataset) * pretrain_cfg.block_size
    print(f"Response tokens (loss active): {int(total_tokens)} / {total_positions} positions")

    # 4. Fine-tune
    trainer = SFTTrainer(model, dataset, tokenizer, cfg, pretrain_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()

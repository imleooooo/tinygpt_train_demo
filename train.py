"""Entry point for training TinyGPT on Tiny Shakespeare."""

import torch

from config import TrainConfig
from src.dataset import TextDataset, download_shakespeare
from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from src.trainer import Trainer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    cfg = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    # 1. Download data
    download_shakespeare(cfg.data_url, cfg.data_file)

    # 2. Load text and build tokenizer
    with open(cfg.data_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset size: {len(text):,} characters")

    tokenizer = CharTokenizer().build(text)
    tokenizer.save(cfg.tokenizer_file)
    print(f"Vocab size: {tokenizer.vocab_size} characters")

    # 3. Encode entire corpus into a flat token tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 4. Train / validation split
    split = int(len(data) * (1 - cfg.val_split))
    train_data = TextDataset(data[:split], cfg.block_size)
    val_data = TextDataset(data[split:], cfg.block_size)
    print(f"Train tokens: {split:,}  |  Val tokens: {len(data) - split:,}")

    # 5. Build model
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    ).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    # 6. Train
    trainer = Trainer(model, train_data, val_data, tokenizer, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()

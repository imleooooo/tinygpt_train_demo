import os

import torch

from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from config import TrainConfig


def load_model(checkpoint_path: str, device: torch.device):
    """Load TinyGPT model and tokenizer from a checkpoint.

    New checkpoints are self-contained: the tokenizer vocab is embedded inside
    them. Old checkpoints (pre-self-contained) fall back to reading
    cfg.tokenizer_file so previously trained checkpoints keep working.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["config"]

    if "tokenizer_char2idx" in ckpt:
        tokenizer = CharTokenizer.from_vocab(ckpt["tokenizer_char2idx"])
    else:
        # Resolve cfg.tokenizer_file relative to the checkpoint's directory so
        # that a moved checkpoint still finds its co-located data/ folder, and
        # cannot accidentally pick up a different run's tokenizer from CWD.
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        tok_path = os.path.join(ckpt_dir, cfg.tokenizer_file)
        tokenizer = CharTokenizer.load(tok_path)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, tokenizer, cfg


def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    # encode() raises ValueError for unseen characters by default, giving a
    # clear message instead of a cryptic IndexError from an empty tensor.
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())

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
        # Two valid legacy layouts exist:
        #   1. checkpoint moved with its data/ folder — tokenizer lives next to it
        #   2. checkpoint saved in a subdir (e.g. runs/run1/) while the tokenizer
        #      remained at the project-root-relative path (e.g. data/tokenizer.json)
        # Try checkpoint-directory-relative first, then CWD-relative, so both
        # layouts keep working without requiring the user to pass extra flags.
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        tok_path_ckpt = os.path.join(ckpt_dir, cfg.tokenizer_file)
        tok_path_cwd = os.path.join(os.getcwd(), cfg.tokenizer_file)
        if os.path.exists(tok_path_ckpt):
            tok_path = tok_path_ckpt
        elif os.path.exists(tok_path_cwd):
            tok_path = tok_path_cwd
        else:
            raise FileNotFoundError(
                f"Cannot locate tokenizer for legacy checkpoint '{checkpoint_path}'. "
                f"Tried:\n  {tok_path_ckpt}\n  {tok_path_cwd}\n"
                "Re-train with the current code to produce a self-contained checkpoint."
            )
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

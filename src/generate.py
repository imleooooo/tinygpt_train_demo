import torch

from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from config import TrainConfig


def load_model(checkpoint_path: str, device: torch.device):
    """Load TinyGPT model and tokenizer from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["config"]

    tokenizer = CharTokenizer.load(cfg.tokenizer_file)

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
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())

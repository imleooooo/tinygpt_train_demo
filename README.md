# TinyGPT Training Demo

A minimal, educational demo that trains a GPT-style language model **from scratch** on Tiny Shakespeare. Runs on CPU or Apple Silicon (MPS) — no GPU required.

## What this is

- **Decoder-only Transformer** (~811K parameters), the same architecture behind GPT-2 and LLaMA
- **Character-level tokenizer** — simple and fully transparent, no external libraries
- **Training objective**: next-token prediction (causal language modeling)
- **Dataset**: Tiny Shakespeare (~1 MB, auto-downloaded)

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (auto-downloads dataset, saves checkpoint.pt when done)
python train.py

# 3. Generate text from the trained model
python generate.py --prompt "ROMEO:\n"
```

Training prints loss every 100 steps and generates a sample every 500 steps so you can watch the model learn.

## Estimated training time

| Hardware | 1 000 steps |
|---|---|
| CPU (Intel) | ~15–20 min |
| CPU (Apple Silicon) | ~8–12 min |
| MPS (Apple Silicon GPU) | ~3–5 min |

## Generation options

```bash
python generate.py --prompt "HAMLET:" --max_new_tokens 300
python generate.py --prompt "ROMEO:\n" --temperature 0.9 --top_k 50
python generate.py --checkpoint checkpoint.pt --prompt "To be"
```

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `"ROMEO:\n"` | Seed text for generation |
| `--max_new_tokens` | `500` | Number of characters to generate |
| `--temperature` | `0.8` | Higher = more random, lower = more repetitive |
| `--top_k` | `40` | Sample from top-k most likely characters |
| `--checkpoint` | `checkpoint.pt` | Path to a saved checkpoint |

> **Note**: prompts may only contain characters present in the training corpus (standard ASCII punctuation and letters). Unseen characters will raise a `ValueError` with a clear message.

## Tuning the model

All hyperparameters live in `config.py`:

```python
@dataclass
class TrainConfig:
    block_size: int = 128   # context window length
    n_embd:     int = 128   # embedding dimension
    n_head:     int = 4     # attention heads
    n_layer:    int = 4     # transformer blocks
    max_iters:  int = 1000  # training steps
    ...
```

Increase `n_embd` / `n_layer` for a larger model, or `max_iters` for longer training.

## Project structure

```
├── config.py          # All hyperparameters
├── train.py           # Training entry point
├── generate.py        # Inference entry point
├── requirements.txt
└── src/
    ├── tokenizer.py   # Character-level tokenizer
    ├── dataset.py     # Sliding-window dataset + data download
    ├── model.py       # CausalSelfAttention, TransformerBlock, TinyGPT
    ├── trainer.py     # Training loop, checkpointing, sampling
    └── generate.py    # load_model() and generate_text() utilities
```

## Checkpoint portability

Checkpoints (`checkpoint.pt`) are self-contained — the tokenizer vocabulary is embedded inside them. You can copy a checkpoint to another machine and run `generate.py --checkpoint /path/to/checkpoint.pt` without needing the original `data/` directory.

# TinyGPT Training Demo

A minimal, educational demo that trains a GPT-style language model **from scratch** on Tiny Shakespeare. Runs on CPU or Apple Silicon (MPS) — no GPU required.

## What this is

This demo covers the first two stages of modern LLM training:

| Stage | What the model learns | Entry point |
|---|---|---|
| **Pre-training** | Next-token prediction on raw text | `train.py` |
| **SFT** | Follow instruction → response format | `sft.py` |

- **Decoder-only Transformer** (~818K parameters), the same architecture behind GPT-2 and LLaMA
- **Character-level tokenizer** — simple and fully transparent, no external libraries
- **Dataset**: Tiny Shakespeare (~1 MB, auto-downloaded) for pre-training; 30 hand-crafted Q&A pairs for SFT

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

## SFT (Supervised Fine-Tuning)

SFT teaches the pretrained model to follow instructions instead of just continuing text.
The key difference from pre-training: **loss is computed only on response tokens** — the
instruction portion contributes zero gradient.

```bash
# 1. Pre-train first (produces checkpoint.pt)
python train.py

# 2. Fine-tune on instruction-response pairs (produces sft_checkpoint.pt)
python sft.py

# 3. Query the fine-tuned model
python generate.py --checkpoint sft_checkpoint.pt \
  --prompt $'INSTRUCTION:\nWho is Romeo?\nRESPONSE:\n'
```

SFT hyperparameters live in `sft_config.py`. Training data is in `data/sft_data.json`
(30 Shakespeare-style Q&A pairs). Extend it with your own examples in the same format:

```json
{"instruction": "your question", "response": "the answer"}
```

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
├── config.py          # Pre-training hyperparameters
├── sft_config.py      # SFT hyperparameters
├── train.py           # Pre-training entry point
├── sft.py             # SFT entry point
├── generate.py        # Inference entry point (works with both checkpoints)
├── requirements.txt
├── data/
│   ├── tiny_shakespeare.txt   # Auto-downloaded
│   └── sft_data.json          # 30 instruction-response pairs
└── src/
    ├── tokenizer.py       # Character-level tokenizer
    ├── dataset.py         # Sliding-window dataset + data download
    ├── model.py           # CausalSelfAttention, TransformerBlock, TinyGPT
    ├── trainer.py         # Pre-training loop, checkpointing, sampling
    ├── sft_dataset.py     # SFT dataset with response-only loss masking
    ├── sft_trainer.py     # SFT training loop
    └── generate.py        # load_model() and generate_text() utilities
```

## Checkpoint portability

Checkpoints (`checkpoint.pt`) are self-contained — the tokenizer vocabulary is embedded inside them. You can copy a checkpoint to another machine and run `generate.py --checkpoint /path/to/checkpoint.pt` without needing the original `data/` directory.

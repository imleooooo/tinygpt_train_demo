# TinyGPT Training Demo

A minimal, educational demo that implements the **full three-stage LLM training pipeline** from scratch on Tiny Shakespeare. Runs on CPU or Apple Silicon (MPS) — no GPU required.

## What this is

| Stage | What the model learns | Entry point | Output |
|---|---|---|---|
| **Pre-training** | Next-token prediction on raw text | `train.py` | `checkpoint.pt` |
| **SFT** | Follow instruction → response format | `sft.py` | `sft_checkpoint.pt` |
| **GRPO** | Maximise a reward signal (alignment) | `grpo.py` | `grpo_checkpoint.pt` |

- **Decoder-only Transformer** (~818K parameters), the same architecture behind GPT-2 and LLaMA
- **Character-level tokenizer** — simple and fully transparent, no external libraries
- **Dataset**: Tiny Shakespeare (~1 MB, auto-downloaded) for pre-training; hand-crafted Q&A pairs for SFT/GRPO

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1 — Pre-train from scratch (~5 min on Apple Silicon)
python train.py

# Stage 2 — Supervised Fine-Tuning
python sft.py

# Stage 3 — GRPO alignment
python grpo.py

# Generate from any checkpoint
python generate.py --checkpoint grpo_checkpoint.pt \
  --prompt $'INSTRUCTION:\nWho is Romeo?\nRESPONSE:\n'
```

---

## Stage 1: Pre-Training

Trains the model to predict the next character on Tiny Shakespeare (~1M characters).
Loss is computed on every token.

```bash
python train.py          # → checkpoint.pt
python generate.py --prompt "ROMEO:"
```

Prints train/val loss every 100 steps and generates a sample every 500 steps.

Key hyperparameters in `config.py`:

```python
block_size: int = 128   # context window length
n_embd:     int = 128   # embedding dimension
n_head:     int = 4     # attention heads
n_layer:    int = 4     # transformer blocks
max_iters:  int = 1000
```

### Estimated pre-training time

| Hardware | 1 000 steps |
|---|---|
| CPU (Intel) | ~15–20 min |
| CPU (Apple Silicon) | ~8–12 min |
| MPS (Apple Silicon GPU) | ~3–5 min |

---

## Stage 2: SFT (Supervised Fine-Tuning)

Teaches the model to follow instructions. The key difference from pre-training:
**loss is computed only on response tokens** — instruction tokens contribute zero gradient.

```bash
python sft.py            # → sft_checkpoint.pt
python generate.py --checkpoint sft_checkpoint.pt \
  --prompt $'INSTRUCTION:\nWho is Hamlet?\nRESPONSE:\n'
```

Training data is in `data/sft_data.json` (30 Shakespeare Q&A pairs). Add your own:

```json
{"instruction": "your question", "response": "the answer"}
```

> Characters must be in the Shakespeare training vocab (standard ASCII letters and punctuation). Unseen characters raise a `ValueError`.

Key hyperparameters in `sft_config.py`: `lr=1e-4`, `batch_size=8`, `max_iters=200`.

---

## Stage 3: GRPO (Group Relative Policy Optimization)

Alignment stage using the algorithm from DeepSeek-R1. GRPO requires **no value/critic network** — it generates G responses per prompt, scores them with a reward function, normalises advantages within the group, and updates the policy with a policy-gradient loss plus a KL penalty against the frozen SFT reference model.

```
L_GRPO = -E[ Σᵢ Aᵢ · log π_θ(yᵢ|x) ] + β · KL(π_θ ‖ π_ref)

Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)    # group-normalised advantage
```

```bash
python grpo.py           # → grpo_checkpoint.pt
python generate.py --checkpoint grpo_checkpoint.pt \
  --prompt $'INSTRUCTION:\nWho is Macbeth?\nRESPONSE:\n'
```

The reward function (`src/reward.py`) is rule-based for demo purposes:

| Signal | Points |
|---|---|
| Response length 40–200 chars | +0.4 |
| Mentions a Shakespeare character name | +0.3 |
| Starts with a capital letter | +0.3 |

In production, swap `compute_reward()` for a learned reward model trained on human preference pairs.

**Implementation notes:**

- **Log-probability**: responses that overflow `block_size` are scored with a batched second pass using the same B-token context window that `generate()` used — so training and sampling see identical conditional distributions.
- **KL penalty**: applied only to the logit rows that predict generated tokens (`[response_start-1, T-1)`), not to the fixed instruction prefix. Logit row `i` predicts token `i+1`, so row `response_start-1` is the first action.
- **Reference model**: the frozen SFT checkpoint loaded with `dropout=0` and `requires_grad=False`. Policy is loaded with the original SFT dropout to avoid over-fitting the small 20-prompt dataset.

Training prints `loss / mean_reward ± std / kl` every 10 steps.

Key hyperparameters in `grpo_config.py`: `G=4`, `beta=0.04`, `lr=5e-5`, `max_iters=100`.

---

## Generation options

```bash
python generate.py --checkpoint <path> --prompt <text>
```

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `"ROMEO:\n"` | Seed text |
| `--max_new_tokens` | `500` | Characters to generate |
| `--temperature` | `0.8` | Higher = more random |
| `--top_k` | `40` | Sample from top-k candidates |
| `--checkpoint` | `checkpoint.pt` | Any stage checkpoint |

All three checkpoints are compatible with `generate.py`.

---

## Project structure

```
├── config.py          # Pre-training hyperparameters
├── sft_config.py      # SFT hyperparameters
├── grpo_config.py     # GRPO hyperparameters
├── train.py           # Stage 1 entry point
├── sft.py             # Stage 2 entry point
├── grpo.py            # Stage 3 entry point
├── generate.py        # Inference (works with any checkpoint)
├── requirements.txt
├── data/
│   ├── tiny_shakespeare.txt   # Auto-downloaded
│   ├── sft_data.json          # 30 instruction-response pairs
│   └── grpo_prompts.json      # 20 instruction prompts for rollouts
└── src/
    ├── tokenizer.py       # Character-level tokenizer
    ├── dataset.py         # Pre-training dataset + data download
    ├── model.py           # CausalSelfAttention, TransformerBlock, TinyGPT
    ├── trainer.py         # Pre-training loop
    ├── sft_dataset.py     # SFT dataset with response-only loss masking
    ├── sft_trainer.py     # SFT training loop
    ├── reward.py          # Rule-based reward function for GRPO
    ├── grpo_trainer.py    # GRPO training loop (rollout, advantages, PG loss)
    └── generate.py        # load_model() and generate_text() utilities
```

## Checkpoint portability

All checkpoints are self-contained — the tokenizer vocabulary is embedded inside them. Copy a checkpoint anywhere and run `generate.py --checkpoint /path/to/checkpoint.pt` without needing the original `data/` directory.

"""GRPO trainer: Group Relative Policy Optimization (DeepSeek-R1 style)."""

import dataclasses
import json
import logging
import time
from typing import NamedTuple

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grpo_config import GRPOConfig
from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from src.reward import compute_reward
from src.sft_dataset import INSTRUCTION_PREFIX, RESPONSE_PREFIX


class Rollout(NamedTuple):
    full_ids: torch.Tensor   # (T,) — prompt + response token IDs
    response_start: int      # index where response tokens begin in full_ids
    reward: float


class GRPOTrainer:
    def __init__(
        self,
        policy: TinyGPT,
        reference: TinyGPT,
        prompts: list[str],
        tokenizer: CharTokenizer,
        config: GRPOConfig,
        pretrain_config,
        device: torch.device,
    ):
        self.policy = policy
        self.reference = reference
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.config = config
        self.pretrain_config = pretrain_config
        self.device = device
        self.block_size = pretrain_config.block_size

        self._wandb_run = None
        self._wandb_run_owned = False
        if config.use_wandb:
            if _wandb is None:
                raise ImportError("wandb is not installed; run `pip install wandb`")
            _pre_existing = _wandb.run is not None
            self._wandb_run = _wandb.init(project="tinygpt-grpo", config=dataclasses.asdict(config))
            self._wandb_run_owned = not _pre_existing

        decay = [p for n, p in policy.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in policy.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _rollout_group(self, prompt: str) -> list[Rollout]:
        """Generate G responses for one prompt and score them."""
        cfg = self.config
        prefix = INSTRUCTION_PREFIX + prompt + RESPONSE_PREFIX
        prefix_ids = self.tokenizer.encode(prefix, errors="ignore") or [0]
        response_start = len(prefix_ids)

        # Crop prefix if it already fills the context
        max_prefix = self.block_size - 1
        prefix_ids = prefix_ids[-max_prefix:]
        response_start = min(response_start, max_prefix)

        prompt_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

        rollouts = []
        self.policy.eval()
        for _ in range(cfg.G):
            out = self.policy.generate(
                prompt_tensor,
                max_new_tokens=cfg.max_gen_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )  # (1, T)
            full_ids = out[0]  # (T,)

            # Decode only the response portion for reward
            response_tokens = full_ids[response_start:].tolist()
            response_text = self.tokenizer.decode(response_tokens)
            reward = compute_reward(response_text)

            rollouts.append(Rollout(full_ids, response_start, reward))
        self.policy.train()
        return rollouts

    # ------------------------------------------------------------------ #
    # Log-probability of response tokens                                  #
    # ------------------------------------------------------------------ #

    def _response_log_prob(self, model: TinyGPT, full_ids: torch.Tensor, response_start: int) -> torch.Tensor:
        """Scalar: sum of log-probs over ALL response tokens under `model`.

        Handles sequences longer than block_size without truncation:

        Part 1 — one forward pass on full_ids[:block_size].
                 Covers response positions [response_start, min(T, block_size)).
                 logits[i] predicts full_ids[i+1], so target at position t
                 uses logit index t-1.

        Part 2 — batched forward pass for overflow positions [block_size, T).
                 generate() crops context with idx[:, -block_size:], so token t
                 is sampled from full_ids[t-block_size : t]  (length block_size).
                 Training must use the same B-token context; the last logit of the
                 forward pass on that window predicts full_ids[t].
                 All such windows are stacked into one batch (T-B, B).
        """
        T = full_ids.shape[0]
        B = self.block_size
        seq_lp = torch.tensor(0.0, device=self.device)

        # Part 1: positions in [response_start, min(T, B))
        end1 = min(T, B)
        if response_start < end1:
            ids1 = full_ids[:end1].unsqueeze(0)           # (1, end1)
            lp1 = F.log_softmax(model(ids1)[0], dim=-1)   # (end1, V)
            # lp1[t-1] predicts full_ids[t]
            tok_lp = lp1[response_start - 1: end1 - 1].gather(
                1, full_ids[response_start:end1].unsqueeze(1)
            ).squeeze(1)
            seq_lp = seq_lp + tok_lp.sum()

        # Part 2: overflow positions [B, T) — each needs its own context window
        if T > B:
            # generate() uses idx[:, -block_size:] to predict the next token,
            # so token at absolute position t is sampled from the distribution
            # produced by context full_ids[t-B : t]  (exactly B tokens).
            # Using B-1 tokens would be a different conditional than the one
            # that produced the action.
            windows = torch.stack(
                [full_ids[t - B: t] for t in range(B, T)]
            )                                              # (T-B, B)
            lp2 = F.log_softmax(model(windows)[:, -1, :], dim=-1)  # (T-B, V)
            targets = full_ids[B:T]                        # (T-B,)
            seq_lp = seq_lp + lp2.gather(1, targets.unsqueeze(1)).squeeze(1).sum()

        return seq_lp

    # ------------------------------------------------------------------ #
    # KL penalty  KL(π_θ ‖ π_ref) averaged over all positions            #
    # ------------------------------------------------------------------ #

    def _kl_penalty(self, full_ids: torch.Tensor, response_start: int) -> torch.Tensor:
        """KL(π_θ ‖ π_ref) averaged over generated response positions only.

        Prompt tokens are excluded: the policy never chose them, so including
        them in the KL budget would incorrectly penalise divergence on the
        fixed instruction prefix and bias training toward matching the reference
        on tokens that were never sampled or rewarded.
        """
        B = self.block_size
        ids = full_ids[:B].unsqueeze(0)   # (1, T_local) where T_local = min(T, B)
        T_local = ids.shape[1]
        rs = min(response_start, T_local)

        with torch.no_grad():
            ref_logits = self.reference(ids)
        pol_logits = self.policy(ids)

        ref_lp = F.log_softmax(ref_logits[0], dim=-1)   # (T_local, V)
        pol_lp = F.log_softmax(pol_logits[0], dim=-1)
        pol_p  = pol_lp.exp()

        # logit row i predicts token i+1, so logit rows [rs-1, T_local-1)
        # are the distributions over the response tokens [rs, T_local).
        # Slicing from rs (the previous off-by-one) would drop the first
        # generated token and include a logit predicting beyond the sequence.
        if rs < 1 or rs >= T_local:
            return torch.tensor(0.0, device=self.device)
        kl_per_token = (pol_p[rs - 1: T_local - 1] * (pol_lp[rs - 1: T_local - 1] - ref_lp[rs - 1: T_local - 1])).sum(-1)
        return kl_per_token.mean() if kl_per_token.numel() > 0 else torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        cfg = self.config
        t0 = time.time()
        prompt_idx = 0

        for step in range(1, cfg.max_iters + 1):
            # Pick a mini-batch of prompts (cycle)
            batch_prompts = [
                self.prompts[(prompt_idx + i) % len(self.prompts)]
                for i in range(cfg.batch_size)
            ]
            prompt_idx = (prompt_idx + cfg.batch_size) % len(self.prompts)

            pg_loss_total = torch.tensor(0.0, device=self.device)
            kl_total      = torch.tensor(0.0, device=self.device)
            all_rewards: list[float] = []

            for prompt in batch_prompts:
                rollouts = self._rollout_group(prompt)
                rewards  = [r.reward for r in rollouts]
                all_rewards.extend(rewards)

                mean_r = sum(rewards) / len(rewards)
                std_r  = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

                for rollout in rollouts:
                    A_i = (rollout.reward - mean_r) / (std_r + 1e-8)
                    lp  = self._response_log_prob(self.policy, rollout.full_ids, rollout.response_start)
                    pg_loss_total = pg_loss_total - A_i * lp
                    kl_total = kl_total + self._kl_penalty(rollout.full_ids, rollout.response_start)

            n = cfg.batch_size * cfg.G
            loss = pg_loss_total / n + cfg.beta * (kl_total / n)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                mean_r = sum(all_rewards) / len(all_rewards)
                std_r  = (sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
                elapsed = time.time() - t0
                logger.info(
                    "step %4d/%d | loss %.4f | reward %.3f ± %.3f | kl %.4f | %.1fs",
                    step, cfg.max_iters, loss.item(), mean_r, std_r,
                    (kl_total / n).item(), elapsed,
                )
                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "reward_mean": mean_r,
                    "reward_std": std_r,
                    "kl": (kl_total / n).item(),
                }
                if self._wandb_run:
                    self._wandb_run.log(metrics)
                if cfg.metrics_file:
                    with open(cfg.metrics_file, "a") as _f:
                        _f.write(json.dumps(metrics) + "\n")
                t0 = time.time()

            if step % cfg.sample_interval == 0:
                self._print_sample("Who is Romeo?")
                self._save_checkpoint(step)

        self._save_checkpoint(cfg.max_iters)
        logger.info("GRPO complete. Checkpoint saved to %s", cfg.grpo_checkpoint)
        if self._wandb_run and self._wandb_run_owned:
            self._wandb_run.finish()

    @torch.no_grad()
    def _print_sample(self, prompt: str) -> None:
        prefix = INSTRUCTION_PREFIX + prompt + RESPONSE_PREFIX
        ids = self.tokenizer.encode(prefix, errors="ignore") or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.policy.eval()
        out = self.policy.generate(idx, self.config.sample_length,
                                   self.config.temperature, self.config.top_k)
        self.policy.train()
        logger.info("Sample output:\n%s", self.tokenizer.decode(out[0].tolist()))

    def _save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "step": step,
                "grpo": True,
                "config": dataclasses.asdict(self.pretrain_config),   # TrainConfig — for load_model() compat
                "grpo_config": dataclasses.asdict(self.config),
                "model_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "tokenizer_char2idx": self.tokenizer.char2idx,
            },
            self.config.grpo_checkpoint,
        )

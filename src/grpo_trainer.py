"""GRPO trainer: Group Relative Policy Optimization (DeepSeek-R1 style)."""

import time
from typing import NamedTuple

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
        """Scalar: sum of log-probs over response tokens under `model`."""
        # full_ids: (T,) — add batch dim
        ids = full_ids.unsqueeze(0)[:, :self.block_size]  # (1, T)
        logits = model(ids)                                # (1, T, V)
        log_probs = F.log_softmax(logits, dim=-1)          # (1, T, V)

        # Positions that predict response tokens:
        # position i in logits predicts token i+1, so response token at
        # position response_start is predicted by logit at response_start-1.
        rs = min(response_start, ids.shape[1] - 1)
        # label ids: response tokens clipped to block_size
        label_ids = ids[0, rs:]         # (L,)
        pred_lp   = log_probs[0, rs - 1: rs - 1 + label_ids.shape[0], :]  # (L, V)
        token_lp  = pred_lp.gather(1, label_ids.unsqueeze(1)).squeeze(1)   # (L,)
        return token_lp.sum()

    # ------------------------------------------------------------------ #
    # KL penalty  KL(π_θ ‖ π_ref) averaged over all positions            #
    # ------------------------------------------------------------------ #

    def _kl_penalty(self, full_ids: torch.Tensor) -> torch.Tensor:
        ids = full_ids.unsqueeze(0)[:, :self.block_size]
        with torch.no_grad():
            ref_logits = self.reference(ids)
        pol_logits = self.policy(ids)
        ref_lp = F.log_softmax(ref_logits, dim=-1)
        pol_lp = F.log_softmax(pol_logits, dim=-1)
        pol_p  = pol_lp.exp()
        kl = (pol_p * (pol_lp - ref_lp)).sum(-1).mean()
        return kl

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
                    kl_total = kl_total + self._kl_penalty(rollout.full_ids)

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
                print(
                    f"step {step:4d}/{cfg.max_iters} | "
                    f"loss {loss.item():.4f} | "
                    f"reward {mean_r:.3f} ± {std_r:.3f} | "
                    f"kl {(kl_total / n).item():.4f} | "
                    f"{elapsed:.1f}s"
                )
                t0 = time.time()

            if step % cfg.sample_interval == 0:
                self._print_sample("Who is Romeo?")
                self._save_checkpoint(step)

        self._save_checkpoint(cfg.max_iters)
        print(f"\nGRPO complete. Checkpoint saved to {cfg.grpo_checkpoint}")

    @torch.no_grad()
    def _print_sample(self, prompt: str) -> None:
        prefix = INSTRUCTION_PREFIX + prompt + RESPONSE_PREFIX
        ids = self.tokenizer.encode(prefix, errors="ignore") or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.policy.eval()
        out = self.policy.generate(idx, self.config.sample_length,
                                   self.config.temperature, self.config.top_k)
        self.policy.train()
        print("\n--- Sample ---")
        print(self.tokenizer.decode(out[0].tolist()))
        print("--------------\n")

    def _save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "step": step,
                "grpo": True,
                "config": self.pretrain_config,   # TrainConfig — for load_model() compat
                "grpo_config": self.config,
                "model_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "tokenizer_char2idx": self.tokenizer.char2idx,
            },
            self.config.grpo_checkpoint,
        )

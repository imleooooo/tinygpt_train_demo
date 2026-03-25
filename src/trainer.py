import dataclasses
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import TrainConfig
from src.model import TinyGPT
from src.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TinyGPT,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: CharTokenizer,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Separate weight decay params from no-decay params (bias, LayerNorm)
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
        )

    @torch.no_grad()
    def _estimate_val_loss(self) -> float:
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True)
        losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            losses.append(loss.item())
            if len(losses) >= 10:  # cap at 10 batches for speed
                break
        self.model.train()
        return sum(losses) / len(losses)

    def _generate_sample(self, prompt: str = "\n") -> str:
        # Use errors="ignore" so unknown characters in the seed prompt are
        # silently dropped rather than crashing training on non-Shakespeare corpora.
        ids = self.tokenizer.encode(prompt, errors="ignore") or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=self.config.sample_length,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )
        self.model.train()
        return self.tokenizer.decode(out[0].tolist())

    def train(self) -> None:
        cfg = self.config
        loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        data_iter = iter(loader)

        self.model.train()
        t0 = time.time()

        for step in range(1, cfg.max_iters + 1):
            # Fetch next batch (cycle through dataset)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Forward + loss
            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                val_loss = self._estimate_val_loss()
                logger.info(
                    "step %5d/%d | train loss %.4f | val loss %.4f | %.1fs elapsed",
                    step, cfg.max_iters, loss.item(), val_loss, elapsed,
                )
                t0 = time.time()

            if step % cfg.sample_interval == 0:
                logger.info("Sample output:\n%s", self._generate_sample("ROMEO:\n"))
                self._save_checkpoint(step)

        # Final checkpoint
        self._save_checkpoint(cfg.max_iters)
        logger.info("Training complete. Checkpoint saved to %s", cfg.checkpoint_file)

    def _save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "step": step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": dataclasses.asdict(self.config),
                "tokenizer_char2idx": self.tokenizer.char2idx,
            },
            self.config.checkpoint_file,
        )

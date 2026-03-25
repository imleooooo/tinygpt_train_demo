import dataclasses
import json
import logging
import time

import torch

try:
    import wandb as _wandb
except ImportError:
    _wandb = None
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sft_config import SFTConfig
from src.model import TinyGPT
from src.tokenizer import CharTokenizer
from src.sft_dataset import INSTRUCTION_PREFIX, RESPONSE_PREFIX

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Trainer for supervised fine-tuning with response-only loss masking."""

    def __init__(
        self,
        model: TinyGPT,
        train_dataset: Dataset,
        tokenizer: CharTokenizer,
        config: SFTConfig,
        pretrain_config,
        device: torch.device,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.pretrain_config = pretrain_config  # stored in checkpoint for load_model() compat
        self.device = device

        self._wandb_run = None
        self._wandb_run_owned = False
        if config.use_wandb:
            if _wandb is None:
                raise ImportError("wandb is not installed; run `pip install wandb`")
            _pre_existing_run = _wandb.run
            self._wandb_run = _wandb.init(project="tinygpt-sft", config=dataclasses.asdict(config))
            self._wandb_run_owned = self._wandb_run is not _pre_existing_run

        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
        )

    def _generate_sample(self, instruction: str) -> str:
        prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
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
            drop_last=False,
        )
        data_iter = iter(loader)

        self.model.train()
        t0 = time.time()

        for step in range(1, cfg.max_iters + 1):
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, mask = next(data_iter)

            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            # Forward
            logits = self.model(x)  # (B, T, vocab_size)

            # Response-only loss: average only over masked (response) positions
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="none",
            )
            loss = (loss_per_token * mask.view(-1)).sum() / mask.sum().clamp(min=1)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                logger.info(
                    "step %4d/%d | loss %.4f | %.1fs elapsed",
                    step, cfg.max_iters, loss.item(), elapsed,
                )
                metrics = {"step": step, "loss": loss.item()}
                if self._wandb_run:
                    self._wandb_run.log(metrics)
                if cfg.metrics_file:
                    with open(cfg.metrics_file, "a") as _f:
                        _f.write(json.dumps(metrics) + "\n")
                t0 = time.time()

            if step % cfg.sample_interval == 0:
                logger.info("Sample output:\n%s", self._generate_sample("Who is Romeo?"))
                self._save_checkpoint(step)

        self._save_checkpoint(cfg.max_iters)
        logger.info("SFT complete. Checkpoint saved to %s", cfg.sft_checkpoint)
        if self._wandb_run and self._wandb_run_owned:
            self._wandb_run.finish()

    def _save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "step": step,
                "sft": True,
                # load_model() reads "config" for model architecture params (block_size etc.)
                # so we store the pretrain TrainConfig here for compatibility.
                "config": dataclasses.asdict(self.pretrain_config),
                "sft_config": dataclasses.asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "tokenizer_char2idx": self.tokenizer.char2idx,
            },
            self.config.sft_checkpoint,
        )

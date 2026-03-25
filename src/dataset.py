import logging
import os
import urllib.request

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def download_shakespeare(url: str, dest: str) -> None:
    """Download Tiny Shakespeare dataset if not already present."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading dataset to %s ...", dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Download complete.")


class TextDataset(Dataset):
    """Sliding-window dataset for causal language modeling.

    Each sample is (input_ids, target_ids) where target_ids is input_ids
    shifted one position to the right.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

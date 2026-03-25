import json

import torch
from torch.utils.data import Dataset

from src.tokenizer import CharTokenizer

INSTRUCTION_PREFIX = "INSTRUCTION:\n"
RESPONSE_PREFIX = "\nRESPONSE:\n"


def format_example(instruction: str, response: str) -> tuple[str, str]:
    """Return (full_text, response_start_marker) for loss mask computation."""
    full = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX + response
    # The response portion begins right after RESPONSE_PREFIX
    response_start = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    return full, response_start


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with response-only loss masking.

    Each item is a tuple (input_ids, labels, loss_mask) of length block_size.
    - input_ids: token IDs of the formatted example (padded/truncated)
    - labels:    input_ids shifted right by one position (next-token targets)
    - loss_mask: 1 for response tokens, 0 for instruction tokens and padding
    """

    def __init__(self, data_file: str, tokenizer: CharTokenizer, block_size: int):
        with open(data_file, "r", encoding="utf-8") as f:
            examples = json.load(f)

        self.block_size = block_size
        self.items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for ex in examples:
            full_text, response_start = format_example(ex["instruction"], ex["response"])

            # Encode full sequence and the instruction prefix
            full_ids = tokenizer.encode(full_text, errors="ignore")
            prefix_len = len(tokenizer.encode(response_start, errors="ignore"))

            # Truncate to block_size + 1 so we can produce block_size label pairs
            full_ids = full_ids[: block_size + 1]

            # Pad to block_size + 1 with zeros
            pad_len = (block_size + 1) - len(full_ids)
            full_ids = full_ids + [0] * pad_len

            ids = torch.tensor(full_ids, dtype=torch.long)
            x = ids[:block_size]           # input
            y = ids[1: block_size + 1]     # labels (shifted by 1)

            # loss_mask: 1 only where y corresponds to a response token
            # response tokens in y start at position (prefix_len - 1) because
            # y is shifted one step to the right relative to x.
            mask = torch.zeros(block_size, dtype=torch.float)
            response_start_in_y = max(prefix_len - 1, 0)
            # Only mask positions that are not padding
            actual_response_len = len(full_ids) - pad_len - 1  # tokens in y that are real
            mask[response_start_in_y:actual_response_len] = 1.0

            self.items.append((x, y, mask))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

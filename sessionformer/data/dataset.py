from typing import Sequence

import torch
from torch.utils.data import Dataset

from sessionformer.data.vocab import PAD_IDX


class SessionDataset(Dataset):
    """Next-item-at-every-position prediction: input = items[:-1], target =
    items[1:], right-padded to max_seq_len. Sessions longer than
    max_seq_len + 1 keep only their most recent max_seq_len + 1 items."""

    def __init__(self, sessions: Sequence[Sequence[int]], max_seq_len: int = 50):
        self.max_seq_len = max_seq_len
        self.sessions = [s for s in sessions if len(s) >= 2]

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int):
        items = list(self.sessions[idx][-(self.max_seq_len + 1) :])
        input_seq = items[:-1]
        target_seq = items[1:]
        length = len(input_seq)
        pad_len = self.max_seq_len - length

        input_padded = input_seq + [PAD_IDX] * pad_len
        target_padded = target_seq + [PAD_IDX] * pad_len

        return (
            torch.tensor(input_padded, dtype=torch.long),
            torch.tensor(target_padded, dtype=torch.long),
            length,
        )

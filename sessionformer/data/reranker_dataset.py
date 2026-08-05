from typing import Sequence

import torch
from torch.utils.data import Dataset

from sessionformer.data.dataset import SessionDataset


class RerankerDataset(Dataset):
    """One example per session, for listwise (softmax cross-entropy) reranker
    training: candidates = [true target] + mined hard negatives, with the
    target always at index 0. hard_negatives must be aligned with
    base_dataset's iteration order (i.e. produced by mine_hard_negatives on
    the same, unshuffled dataset) and have the same length for every
    session, so candidates can be stacked into a batch tensor."""

    def __init__(self, base_dataset: SessionDataset, hard_negatives: Sequence[Sequence[int]]):
        if len(hard_negatives) != len(base_dataset):
            raise ValueError("hard_negatives must have one entry per session in base_dataset")
        self.base_dataset = base_dataset
        self.hard_negatives = hard_negatives

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        input_seq, target_seq, length = self.base_dataset[idx]
        target = target_seq[length - 1].item()
        candidates = torch.tensor([target, *self.hard_negatives[idx]], dtype=torch.long)
        return input_seq, length, candidates

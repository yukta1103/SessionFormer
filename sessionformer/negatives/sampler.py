import torch

from sessionformer.data.vocab import PAD_IDX, UNK_IDX

FIRST_REAL_ITEM_IDX = max(PAD_IDX, UNK_IDX) + 1


class UniformNegativeSampler:
    """Samples item indices uniformly from the real vocabulary (excluding
    PAD/UNK), resampling any negative that collides with its positive."""

    def __init__(self, vocab_size: int, num_negatives: int = 100):
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives

    def sample(self, positive: torch.Tensor) -> torch.Tensor:
        shape = positive.shape + (self.num_negatives,)
        negatives = torch.randint(FIRST_REAL_ITEM_IDX, self.vocab_size, shape, device=positive.device)
        collision = negatives == positive.unsqueeze(-1)
        while collision.any():
            resample = torch.randint(FIRST_REAL_ITEM_IDX, self.vocab_size, shape, device=positive.device)
            negatives = torch.where(collision, resample, negatives)
            collision = negatives == positive.unsqueeze(-1)
        return negatives

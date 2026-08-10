import torch


def softmax_entropy(scores: torch.Tensor) -> torch.Tensor:
    """scores: (..., vocab_size) logits (invalid items like PAD/UNK should
    already be masked to -inf by the caller) -> (...) entropy of the softmax
    distribution, in nats."""
    probs = torch.softmax(scores, dim=-1)
    return -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)


class EntropyGate:
    """Decides, from SASRec's own prediction entropy, whether its top-K is
    trusted directly or an expensive refiner should be invoked. Deliberately
    knows nothing about what the refiner is (reranker today, potentially a
    critic-stage model later) -- it only answers yes/no from entropy."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_refine(self, entropy: torch.Tensor) -> torch.Tensor:
        return entropy > self.threshold

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import torch.nn.functional as F

from sessionformer.utils.training import train_step


class _StubModel:
    """Deterministic stand-in for a real model: hidden states are just the
    input indices broadcast to a 1-dim feature, and score(hidden, items) is
    a fixed, hand-computable function of (hidden, item index) so the loss
    contributed by each term can be checked exactly."""

    def encode_sequence(self, input_seq: torch.Tensor) -> torch.Tensor:
        return input_seq.unsqueeze(-1).float()

    def score(self, hidden: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # hidden: (..., 1), item_indices: (..., C) -> (..., C) via broadcasting.
        return hidden - item_indices.float()


class _NoOpSampler:
    def __init__(self, negatives: torch.Tensor):
        self._negatives = negatives

    def sample(self, target_seq: torch.Tensor) -> torch.Tensor:
        return self._negatives


def test_self_negative_is_a_harmless_duplicate_for_genuine_repeats():
    model = _StubModel()
    device = torch.device("cpu")

    # input == target everywhere: a real repeat-click at every position.
    # The self-negative must NOT inject input_seq's own (correct) value into
    # the negative pool -- it should fall back to duplicating an existing
    # negative, which changes nothing since averaging a duplicate value
    # leaves the mean unchanged.
    input_seq = torch.tensor([[5, 5]])
    target_seq = torch.tensor([[5, 5]])
    negatives = torch.tensor([[[7], [7]]])
    sampler = _NoOpSampler(negatives)

    loss_with_fix = train_step(model, sampler, input_seq, target_seq, device)

    hidden = model.encode_sequence(input_seq)
    pos_scores = model.score(hidden, target_seq.unsqueeze(-1)).squeeze(-1)
    neg_scores = model.score(hidden, negatives)
    expected = (-F.logsigmoid(pos_scores) - F.logsigmoid(-neg_scores).mean(dim=-1)).mean()

    assert torch.allclose(loss_with_fix, expected, atol=1e-6)


def test_self_negative_is_folded_into_the_pool_for_a_wrong_echo():
    model = _StubModel()
    device = torch.device("cpu")

    # input != target: the wrong echo (5) must be folded into the averaged
    # negative pool alongside the sampled negative (7), not skipped and not
    # added as a separate full-weight term.
    input_seq = torch.tensor([[5, 5]])
    target_seq = torch.tensor([[9, 9]])
    negatives = torch.tensor([[[7], [7]]])
    sampler = _NoOpSampler(negatives)

    loss_with_fix = train_step(model, sampler, input_seq, target_seq, device)

    hidden = model.encode_sequence(input_seq)
    pos_scores = model.score(hidden, target_seq.unsqueeze(-1)).squeeze(-1)
    all_negatives = torch.tensor([[[7, 5], [7, 5]]])
    neg_scores = model.score(hidden, all_negatives)
    expected = (-F.logsigmoid(pos_scores) - F.logsigmoid(-neg_scores).mean(dim=-1)).mean()

    assert torch.allclose(loss_with_fix, expected, atol=1e-6)

    # And it must differ from what the loss would be if the echo were
    # ignored entirely (the pre-fix behavior), confirming it has real effect.
    neg_scores_without_fix = model.score(hidden, negatives)
    loss_without_fix = (-F.logsigmoid(pos_scores) - F.logsigmoid(-neg_scores_without_fix).mean(dim=-1)).mean()
    assert not torch.allclose(loss_with_fix, loss_without_fix, atol=1e-6)

import torch

from sessionformer.gating.entropy_gate import EntropyGate, softmax_entropy


def test_uniform_distribution_has_higher_entropy_than_peaked():
    uniform_scores = torch.zeros(1, 10)
    peaked_scores = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert softmax_entropy(uniform_scores).item() > softmax_entropy(peaked_scores).item()


def test_masked_items_do_not_contribute_to_entropy():
    scores = torch.tensor([[10.0, 0.0, float("-inf"), float("-inf")]])
    # only two real candidates remain after masking -> entropy should match
    # the two-item equivalent, not be inflated by the masked ones
    two_item_scores = torch.tensor([[10.0, 0.0]])
    assert torch.allclose(softmax_entropy(scores), softmax_entropy(two_item_scores), atol=1e-5)


def test_gate_fires_above_threshold_only():
    gate = EntropyGate(threshold=1.0)
    entropy = torch.tensor([0.5, 1.0, 1.5, 2.0])
    assert gate.should_refine(entropy).tolist() == [False, False, True, True]

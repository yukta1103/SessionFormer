import torch

from sessionformer.data.vocab import PAD_IDX
from sessionformer.models.reranker import Reranker


def _model(**overrides):
    defaults = dict(vocab_size=50, max_seq_len=6, embedding_dim=8, num_heads=2, num_blocks=1, ff_hidden_dim=16)
    defaults.update(overrides)
    return Reranker(**defaults)


def test_build_input_places_candidate_at_true_length_not_padded_end():
    model = _model()
    input_seq = torch.tensor([[5, 6, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]])
    lengths = torch.tensor([2])
    candidate = torch.tensor([99])

    extended, candidate_pos = model.build_input(input_seq, lengths, candidate)

    assert candidate_pos.item() == 2
    assert extended[0, :2].tolist() == [5, 6]
    assert extended[0, 2].item() == 99
    assert extended[0, 3:].tolist() == [PAD_IDX] * 4  # max_seq_len=6, +1 for the candidate slot


def test_forward_output_shape():
    model = _model()
    input_seq = torch.randint(2, 50, (4, 6))
    lengths = torch.tensor([3, 4, 6, 1])
    candidate = torch.randint(2, 50, (4,))

    scores = model(input_seq, lengths, candidate)
    assert scores.shape == (4,)


def test_different_candidates_yield_different_scores():
    model = _model()
    model.eval()
    input_seq = torch.tensor([[5, 6, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]])
    lengths = torch.tensor([2])

    score_a = model(input_seq, lengths, torch.tensor([10]))
    score_b = model(input_seq, lengths, torch.tensor([20]))
    assert not torch.allclose(score_a, score_b)


def test_score_candidates_matches_individual_forward_calls():
    model = _model()
    model.eval()
    input_seq = torch.randint(2, 50, (2, 6))
    lengths = torch.tensor([3, 5])
    candidates = torch.tensor([[10, 20, 30], [15, 25, 35]])

    batched = model.score_candidates(input_seq, lengths, candidates)
    assert batched.shape == (2, 3)

    for i in range(2):
        for j in range(3):
            individual = model(input_seq[i : i + 1], lengths[i : i + 1], candidates[i : i + 1, j])
            assert torch.allclose(batched[i, j], individual[0], atol=1e-5)

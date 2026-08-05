import torch

from sessionformer.models.sasrec import SASRec


def _model(**overrides):
    defaults = dict(vocab_size=50, max_seq_len=10, embedding_dim=8, num_heads=2, num_blocks=2, ff_hidden_dim=16)
    defaults.update(overrides)
    return SASRec(**defaults)


def test_encode_sequence_output_shape():
    model = _model()
    input_seq = torch.randint(0, 50, (4, 6))
    hidden = model.encode_sequence(input_seq)
    assert hidden.shape == (4, 6, 8)


def test_score_output_shape_matches_candidates():
    model = _model()
    input_seq = torch.randint(0, 50, (4, 6))
    hidden = model.encode_sequence(input_seq)
    candidates = torch.randint(0, 50, (4, 6, 10))
    scores = model.score(hidden, candidates)
    assert scores.shape == (4, 6, 10)


def test_score_all_items_output_shape():
    model = _model()
    hidden = torch.randn(4, 8)
    scores = model.score_all_items(hidden)
    assert scores.shape == (4, 50)


def test_encoder_is_causal():
    model = _model()
    model.eval()

    input_seq = torch.randint(1, 50, (1, 6))
    hidden_before = model.encode_sequence(input_seq)

    changed = input_seq.clone()
    changed[0, -1] = (changed[0, -1] % 49) + 1  # perturb only the last position's item
    hidden_after = model.encode_sequence(changed)

    assert torch.allclose(hidden_before[0, :-1], hidden_after[0, :-1], atol=1e-6)

import torch

from sessionformer.models.gru4rec import GRU4Rec


def test_encode_sequence_output_shape():
    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1)
    input_seq = torch.randint(0, 50, (4, 6))
    hidden = model.encode_sequence(input_seq)
    assert hidden.shape == (4, 6, 8)


def test_score_output_shape_matches_candidates():
    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1)
    input_seq = torch.randint(0, 50, (4, 6))
    hidden = model.encode_sequence(input_seq)
    candidates = torch.randint(0, 50, (4, 6, 10))
    scores = model.score(hidden, candidates)
    assert scores.shape == (4, 6, 10)


def test_score_all_items_output_shape():
    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1)
    hidden = torch.randn(4, 8)
    scores = model.score_all_items(hidden)
    assert scores.shape == (4, 50)


def test_hidden_and_output_dim_differ_with_projection():
    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=16, num_layers=1)
    input_seq = torch.randint(0, 50, (2, 5))
    hidden = model.encode_sequence(input_seq)
    assert hidden.shape == (2, 5, 8)  # projected back to embedding_dim

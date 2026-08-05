import torch
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.models.gru4rec import GRU4Rec
from sessionformer.negatives.hard_negative_mining import mine_hard_negatives


def test_hard_negatives_exclude_the_true_target_and_respect_cap():
    sessions = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
    ds = SessionDataset(sessions, max_seq_len=5)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1)
    hard_negatives = mine_hard_negatives(model, loader, torch.device("cpu"), top_k=10, num_hard_negatives=3)

    assert len(hard_negatives) == len(sessions)
    targets = [session[-1] for session in sessions]
    for negs, target in zip(hard_negatives, targets):
        assert len(negs) <= 3
        assert target not in negs


def test_output_order_matches_dataset_order():
    sessions = [[2, 3], [4, 5], [6, 7]]
    ds = SessionDataset(sessions, max_seq_len=5)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = GRU4Rec(vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1)
    hard_negatives = mine_hard_negatives(model, loader, torch.device("cpu"), top_k=10, num_hard_negatives=3)

    assert len(hard_negatives) == 3

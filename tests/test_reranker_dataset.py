from sessionformer.data.dataset import SessionDataset
from sessionformer.data.reranker_dataset import RerankerDataset


def test_one_example_per_session():
    base = SessionDataset([[10, 11, 12], [20, 21]], max_seq_len=5)
    ds = RerankerDataset(base, [[30, 31], [40, 41]])
    assert len(ds) == 2


def test_candidates_are_target_followed_by_hard_negatives():
    base = SessionDataset([[10, 11, 12]], max_seq_len=5)
    ds = RerankerDataset(base, [[30, 31]])

    input_seq, length, candidates = ds[0]
    assert length == 2
    assert candidates.tolist() == [12, 30, 31]  # target (12) first, then hard negatives


def test_mismatched_hard_negatives_length_raises():
    base = SessionDataset([[10, 11, 12], [20, 21]], max_seq_len=5)
    try:
        RerankerDataset(base, [[30]])
        assert False, "expected ValueError"
    except ValueError:
        pass

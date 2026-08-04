from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX


def test_shifts_input_and_target_by_one():
    ds = SessionDataset([[10, 11, 12]], max_seq_len=5)
    input_seq, target_seq, length = ds[0]
    assert input_seq[:2].tolist() == [10, 11]
    assert target_seq[:2].tolist() == [11, 12]
    assert length == 2


def test_pads_short_sessions_with_pad_idx():
    ds = SessionDataset([[10, 11, 12]], max_seq_len=5)
    input_seq, target_seq, length = ds[0]
    assert length == 2
    assert input_seq[2:].tolist() == [PAD_IDX] * 3
    assert target_seq[2:].tolist() == [PAD_IDX] * 3


def test_truncates_long_sessions_to_most_recent_items():
    items = list(range(100, 110))  # 10 items
    ds = SessionDataset([items], max_seq_len=3)
    input_seq, target_seq, length = ds[0]
    assert length == 3
    # last max_seq_len + 1 = 4 items are [106, 107, 108, 109]
    assert input_seq.tolist() == [106, 107, 108]
    assert target_seq.tolist() == [107, 108, 109]


def test_sessions_shorter_than_two_are_dropped():
    ds = SessionDataset([[10, 11], [42]], max_seq_len=5)
    assert len(ds) == 1


def test_dataset_length_matches_kept_sessions():
    ds = SessionDataset([[1, 2], [3, 4, 5], [6]], max_seq_len=5)
    assert len(ds) == 2

from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab


def test_build_assigns_unique_indices_excluding_reserved():
    vocab = ItemVocab.build([10, 20, 30, 10])
    indices = {vocab.encode(i) for i in (10, 20, 30)}
    assert len(indices) == 3
    assert PAD_IDX not in indices
    assert UNK_IDX not in indices


def test_unseen_item_maps_to_unk():
    vocab = ItemVocab.build([10, 20])
    assert vocab.encode(999) == UNK_IDX


def test_encode_decode_roundtrip():
    vocab = ItemVocab.build([10, 20, 30])
    idx = vocab.encode(20)
    assert vocab.decode(idx) == 20


def test_len_includes_pad_and_unk():
    vocab = ItemVocab.build([10, 20, 30])
    assert len(vocab) == 3 + 2


def test_save_and_load_roundtrip(tmp_path):
    vocab = ItemVocab.build([10, 20, 30])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = ItemVocab.load(path)
    assert loaded.encode(20) == vocab.encode(20)
    assert len(loaded) == len(vocab)

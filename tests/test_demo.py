# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch

from app.demo import clamp_step, format_item, get_predictions, predictions_dataframe, score_step
from sessionformer.data.vocab import ItemVocab
from sessionformer.gating.entropy_gate import EntropyGate
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec

MAX_SEQ_LEN = 10


def test_clamp_step_keeps_valid_index_in_range():
    assert clamp_step(2, 5) == 2


def test_clamp_step_caps_at_last_valid_index():
    # e.g. Next was clicked right as a shorter session got shuffled in
    assert clamp_step(5, 3) == 2


def test_clamp_step_floors_at_zero():
    assert clamp_step(-1, 3) == 0


def test_clamp_step_handles_zero_steps_without_going_negative():
    assert clamp_step(0, 0) == 0


def _vocab(n_items: int = 20) -> ItemVocab:
    return ItemVocab.build(range(100, 100 + n_items))


def _models(vocab: ItemVocab):
    sasrec = SASRec(vocab_size=len(vocab), max_seq_len=MAX_SEQ_LEN, embedding_dim=8, num_heads=2, num_blocks=1, ff_hidden_dim=16)
    reranker = Reranker(vocab_size=len(vocab), max_seq_len=MAX_SEQ_LEN, embedding_dim=8, num_heads=2, num_blocks=1, ff_hidden_dim=16)
    return sasrec, reranker


def test_get_predictions_shapes_and_keys_when_gate_always_fires():
    vocab = _vocab()
    sasrec, reranker = _models(vocab)
    gate = EntropyGate(threshold=-1.0)  # always fires -> exercise the reranker branch
    device = torch.device("cpu")

    item_ids = [100, 101, 102, 103, 104]
    steps = get_predictions(sasrec, reranker, gate, vocab, device, MAX_SEQ_LEN, item_ids)

    assert len(steps) == len(item_ids) - 1
    for i, step in enumerate(steps):
        assert step["context_ids"] == item_ids[: i + 1]
        assert step["target_id"] == item_ids[i + 1]
        assert step["gate_fired"] is True
        assert len(step["plain_topk_ids"]) == 10
        assert len(step["plain_topk_probs"]) == 10
        assert len(step["reranked_topk_ids"]) == 10
        assert len(step["reranked_topk_scores"]) == 10
        assert step["final_topk_ids"] == step["reranked_topk_ids"]
        assert step["is_hit"] == (step["target_id"] in step["final_topk_ids"])


def test_get_predictions_skips_reranker_when_gate_never_fires():
    vocab = _vocab()
    sasrec, reranker = _models(vocab)
    gate = EntropyGate(threshold=1e9)  # never fires
    device = torch.device("cpu")

    item_ids = [100, 101, 102]
    steps = get_predictions(sasrec, reranker, gate, vocab, device, MAX_SEQ_LEN, item_ids)

    for step in steps:
        assert step["gate_fired"] is False
        assert step["reranked_topk_ids"] is None
        assert step["reranked_topk_scores"] is None
        assert len(step["plain_topk_probs"]) == 10
        assert step["final_topk_ids"] == step["plain_topk_ids"]
        assert step["is_hit"] == (step["target_id"] in step["plain_topk_ids"])


def test_format_item_without_category():
    assert format_item(123, {}) == "#123"


def test_format_item_with_category():
    assert format_item(123, {123: "42"}) == "#123 (category 42)"


def test_score_step_counts_a_new_step_once():
    scored, hits, total = score_step(True, 0, set(), hits=0, total=0)
    assert scored == {0}
    assert hits == 1
    assert total == 1


def test_score_step_counts_a_miss():
    scored, hits, total = score_step(False, 0, set(), hits=0, total=0)
    assert hits == 0
    assert total == 1


def test_score_step_does_not_double_count_a_revisited_step():
    scored, hits, total = score_step(True, 0, set(), hits=0, total=0)
    scored, hits, total = score_step(True, 0, scored, hits=hits, total=total)
    assert hits == 1
    assert total == 1


def test_predictions_dataframe_labels_the_correct_item():
    df = predictions_dataframe([10, 20, 30], [0.5, 0.3, 0.2], target_id=20, categories={})
    assert df.loc["#20", "type"] == "Correct next item"
    assert df.loc["#10", "type"] == "Other guess"
    assert df.loc["#30", "type"] == "Other guess"


def test_predictions_dataframe_preserves_scores():
    df = predictions_dataframe([10, 20], [0.7, 0.3], target_id=10, categories={})
    assert df.loc["#10", "score"] == 0.7
    assert df.loc["#20", "score"] == 0.3

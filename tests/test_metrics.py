import numpy as np

from eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_hand_computed():
    topk = np.array([[5, 3, 8, 1, 9]])
    target = np.array([3])  # position 1 (0-indexed)
    assert recall_at_k(topk, target, k=5) == 1.0
    assert recall_at_k(topk, target, k=2) == 1.0
    assert recall_at_k(topk, target, k=1) == 0.0


def test_recall_at_k_miss_is_zero():
    topk = np.array([[5, 3, 8, 1, 9]])
    target = np.array([7])
    assert recall_at_k(topk, target, k=5) == 0.0


def test_recall_at_k_averages_across_rows():
    topk = np.array([[5, 3], [8, 1]])
    target = np.array([3, 99])  # hit, miss
    assert recall_at_k(topk, target, k=2) == 0.5


def test_ndcg_at_k_hand_computed():
    topk = np.array([[5, 3, 8, 1, 9]])
    target = np.array([3])  # 0-indexed position 1 -> rank 2 -> 1/log2(3)
    expected = 1.0 / np.log2(3)
    assert abs(ndcg_at_k(topk, target, k=5) - expected) < 1e-9


def test_ndcg_at_k_top_rank_is_one():
    topk = np.array([[3, 5, 8]])
    target = np.array([3])  # rank 1 -> 1/log2(2) == 1.0
    assert abs(ndcg_at_k(topk, target, k=3) - 1.0) < 1e-9


def test_ndcg_at_k_miss_is_zero():
    topk = np.array([[5, 3, 8]])
    target = np.array([99])
    assert ndcg_at_k(topk, target, k=3) == 0.0


def test_mrr_hand_computed():
    topk = np.array([[5, 3, 8, 1, 9], [3, 5, 8, 1, 9]])
    target = np.array([3, 3])  # rank 2 (rr=0.5), rank 1 (rr=1.0)
    assert abs(mrr(topk, target) - 0.75) < 1e-9


def test_mrr_miss_contributes_zero():
    topk = np.array([[5, 3, 8]])
    target = np.array([99])
    assert mrr(topk, target) == 0.0

import numpy as np


def _rank_position(topk: np.ndarray, target: np.ndarray) -> np.ndarray:
    """0-indexed position of target within each row of topk, or -1 if absent."""
    matches = topk == target[:, None]
    has_match = matches.any(axis=1)
    return np.where(has_match, matches.argmax(axis=1), -1)


def recall_at_k(topk: np.ndarray, target: np.ndarray, k: int) -> float:
    hits = (topk[:, :k] == target[:, None]).any(axis=1)
    return float(hits.mean())


def ndcg_at_k(topk: np.ndarray, target: np.ndarray, k: int) -> float:
    position = _rank_position(topk[:, :k], target)
    safe_position = np.maximum(position, 0)  # avoid computing log2 on the sentinel -1
    ndcg = np.where(position >= 0, 1.0 / np.log2(safe_position + 2), 0.0)
    return float(ndcg.mean())


def mrr(topk: np.ndarray, target: np.ndarray) -> float:
    """Reciprocal rank within the provided topk width; a target ranked
    beyond that width contributes 0, same convention as recall/ndcg here."""
    position = _rank_position(topk, target)
    safe_position = np.maximum(position, 0)
    rr = np.where(position >= 0, 1.0 / (safe_position + 1), 0.0)
    return float(rr.mean())

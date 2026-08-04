from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp


class ItemKNNBaseline:
    """Session co-occurrence baseline: recommends items that most often appear
    in the same session as the items already in the current session."""

    def __init__(self):
        self.co_occurrence: Optional[sp.csr_matrix] = None

    def fit(self, sessions: Sequence[Sequence[int]], vocab_size: int) -> "ItemKNNBaseline":
        rows, cols = [], []
        for session_idx, items in enumerate(sessions):
            for item in set(items):
                rows.append(session_idx)
                cols.append(item)

        data = np.ones(len(rows), dtype=np.float32)
        incidence = sp.csr_matrix((data, (rows, cols)), shape=(len(sessions), vocab_size))

        co_occurrence = (incidence.T @ incidence).tocsr()
        co_occurrence.setdiag(0)
        co_occurrence.eliminate_zeros()
        self.co_occurrence = co_occurrence
        return self

    def recommend(self, session_items: Sequence[int], k: int) -> list[int]:
        if self.co_occurrence is None:
            raise RuntimeError("fit() must be called before recommend()")
        if not session_items:
            return []

        exclude = set(session_items)
        scores = np.asarray(self.co_occurrence[list(exclude), :].sum(axis=0)).ravel()
        # Only sort candidates with nonzero co-occurrence instead of the whole
        # vocab — co-occurrence is sparse, so this is typically orders of
        # magnitude smaller than vocab_size.
        candidates = np.flatnonzero(scores)
        if exclude:
            candidates = candidates[~np.isin(candidates, list(exclude))]
        if candidates.size == 0:
            return []

        if candidates.size > k:
            top_k = np.argpartition(-scores[candidates], k)[:k]
            candidates = candidates[top_k]
        order = np.argsort(-scores[candidates])
        return [int(i) for i in candidates[order]]

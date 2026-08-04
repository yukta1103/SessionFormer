import itertools
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


class PopularityBaseline:
    def __init__(self):
        self.ranked_items: Optional[np.ndarray] = None  # item indices, most popular first

    def fit(self, item_indices: Iterable[int]) -> "PopularityBaseline":
        counts = pd.Series(list(item_indices)).value_counts()
        self.ranked_items = counts.index.to_numpy()
        return self

    def recommend(self, session_items: Sequence[int], k: int) -> list[int]:
        if self.ranked_items is None:
            raise RuntimeError("fit() must be called before recommend()")
        exclude = set(session_items)
        candidates = (int(item) for item in self.ranked_items if item not in exclude)
        return list(itertools.islice(candidates, k))

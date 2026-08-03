import json
from pathlib import Path
from typing import Iterable, Optional, Union

PAD_IDX = 0
UNK_IDX = 1


class ItemVocab:
    def __init__(self, item2idx: dict[int, int]):
        self.item2idx = item2idx
        self.idx2item = {idx: item for item, idx in item2idx.items()}

    @classmethod
    def build(cls, item_ids: Iterable[int]) -> "ItemVocab":
        unique_items = sorted(set(int(i) for i in item_ids))
        item2idx = {item: idx + 2 for idx, item in enumerate(unique_items)}
        return cls(item2idx)

    def encode(self, item_id: int) -> int:
        return self.item2idx.get(item_id, UNK_IDX)

    def decode(self, idx: int) -> Optional[int]:
        return self.idx2item.get(idx)

    def __len__(self) -> int:
        return len(self.item2idx) + 2  # + PAD_IDX, UNK_IDX

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps({str(k): v for k, v in self.item2idx.items()}))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ItemVocab":
        raw = json.loads(Path(path).read_text())
        return cls({int(k): v for k, v in raw.items()})

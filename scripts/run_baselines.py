from pathlib import Path

import pandas as pd

from sessionformer.baselines.item_knn import ItemKNNBaseline
from sessionformer.baselines.popularity import PopularityBaseline
from sessionformer.data.vocab import ItemVocab


def _leave_one_out_recall_at_k(model, sessions_by_id: dict, k: int) -> float:
    """Informal sanity check, not the formal Phase 8 evaluation: holds out the
    last item of each session as the target and checks if it lands in top-k."""
    hits, total = 0, 0
    for items in sessions_by_id.values():
        if len(items) < 2:
            continue
        context, target = items[:-1], items[-1]
        recs = model.recommend(context, k)
        hits += int(target in recs)
        total += 1
    return hits / total if total else 0.0


def main() -> None:
    processed_dir = Path("data/processed")
    sessions = pd.read_parquet(processed_dir / "sessions.parquet")
    vocab = ItemVocab.load(processed_dir / "vocab.json")

    sessions["item_idx"] = sessions["itemid"].map(vocab.encode)

    train = sessions[sessions["split"] == "train"]
    val = sessions[sessions["split"] == "val"]

    train_sessions_by_id = train.groupby("session_id")["item_idx"].apply(list).to_dict()
    val_sessions_by_id = val.groupby("session_id")["item_idx"].apply(list).to_dict()

    popularity = PopularityBaseline().fit(train["item_idx"])
    item_knn = ItemKNNBaseline().fit(list(train_sessions_by_id.values()), vocab_size=len(vocab))

    print(f"Fitted on {len(train_sessions_by_id)} train sessions, vocab size {len(vocab)}")
    for name, model in [("Popularity", popularity), ("Item-KNN", item_knn)]:
        recall10 = _leave_one_out_recall_at_k(model, val_sessions_by_id, k=10)
        print(f"{name}: Recall@10 (val, leave-one-out sanity check) = {recall10:.4f}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from sessionformer.data.sessionize import sessionize
from sessionformer.data.splits import assign_split, identify_cold_start_items, identify_cold_start_users
from sessionformer.data.vocab import ItemVocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    events = pd.read_csv(Path(args.data_dir) / "events.csv")
    sessions = sessionize(
        events,
        timeout_seconds=cfg["session_timeout_minutes"] * 60,
        min_session_length=cfg["min_session_length"],
    )

    session_starts = sessions.groupby("session_id")["timestamp"].min()
    split_by_session = assign_split(
        session_starts,
        val_start_date=cfg["val_start_date"],
        test_start_date=cfg["test_start_date"],
        test_end_date=cfg["test_end_date"],
    )
    sessions["split"] = sessions["session_id"].map(split_by_session)

    train_events = sessions[sessions["split"] == "train"]
    vocab = ItemVocab.build(train_events["itemid"])

    cold_items = identify_cold_start_items(train_events, cfg["cold_start_item_min_train_interactions"])
    cold_users = identify_cold_start_users(train_events, cfg["cold_start_user_min_train_interactions"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(out_dir / "sessions.parquet", index=False)
    vocab.save(out_dir / "vocab.json")
    (out_dir / "cold_start_items.json").write_text(json.dumps(sorted(cold_items)))
    (out_dir / "cold_start_users.json").write_text(json.dumps(sorted(cold_users)))

    split_counts = sessions.groupby("split")["session_id"].nunique()
    print(f"Total sessions: {sessions['session_id'].nunique()}")
    print(f"Sessions per split:\n{split_counts}")
    print(f"Vocab size (incl. PAD/UNK): {len(vocab)}")
    print(f"Cold-start items (train): {len(cold_items)}")
    print(f"Cold-start users (train): {len(cold_users)}")


if __name__ == "__main__":
    main()

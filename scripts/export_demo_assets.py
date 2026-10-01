import argparse
import json
from pathlib import Path

import pandas as pd

MIN_LEN, MAX_LEN = 4, 15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = pd.read_parquet(data_dir / "sessions.parquet")
    test = sessions[sessions["split"] == "test"]
    grouped = test.groupby("session_id")["itemid"].apply(list)
    demo_sessions = [items for items in grouped if MIN_LEN <= len(items) <= MAX_LEN]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo_sessions.json").write_text(json.dumps(demo_sessions))

    item_ids_in_demo = {item_id for session in demo_sessions for item_id in session}
    full_categories = json.loads((data_dir / "item_categories.json").read_text())
    demo_categories = {k: v for k, v in full_categories.items() if int(k) in item_ids_in_demo}
    (out_dir / "demo_item_categories.json").write_text(json.dumps(demo_categories))

    print(f"demo_sessions.json: {len(demo_sessions)} sessions, {len(item_ids_in_demo)} distinct items")
    print(f"demo_item_categories.json: {len(demo_categories)} entries")


if __name__ == "__main__":
    main()

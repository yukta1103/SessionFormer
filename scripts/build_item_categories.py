import argparse
import json
from pathlib import Path

import pandas as pd

from sessionformer.content.item_text import build_item_categories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parts = [pd.read_csv(data_dir / f"item_properties_part{i}.csv") for i in (1, 2)]
    item_properties = pd.concat(parts, ignore_index=True)

    categories = build_item_categories(item_properties)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "item_categories.json").write_text(json.dumps({str(k): v for k, v in categories.items()}))

    print(f"Built category lookup for {len(categories)} items")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import pandas as pd

from sessionformer.content.item_text import build_item_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parts = [pd.read_csv(data_dir / f"item_properties_part{i}.csv") for i in (1, 2)]
    item_properties = pd.concat(parts, ignore_index=True)

    item_text = build_item_text(item_properties)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "item_text.json").write_text(json.dumps({str(k): v for k, v in item_text.items()}))

    print(f"Built pseudo-documents for {len(item_text)} items")
    sample_id = next(iter(item_text))
    print(f"Sample (itemid={sample_id}): {item_text[sample_id][:200]}...")


if __name__ == "__main__":
    main()

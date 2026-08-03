import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/sample")
    parser.add_argument("--n-visitors", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "events.csv")
    rng = np.random.default_rng(args.seed)
    visitors = events["visitorid"].unique()
    sampled_visitors = rng.choice(visitors, size=args.n_visitors, replace=False)

    sample_events = events[events["visitorid"].isin(sampled_visitors)].sort_values(["visitorid", "timestamp"])
    sample_events.to_csv(out_dir / "events.csv", index=False)

    sample_items = set(sample_events["itemid"].unique())

    for part in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        props = pd.read_csv(data_dir / part)
        props = props[props["itemid"].isin(sample_items)]
        props.to_csv(out_dir / part, index=False)

    tree = pd.read_csv(data_dir / "category_tree.csv")
    tree.to_csv(out_dir / "category_tree.csv", index=False)

    print(f"Sample: {len(sample_events)} events, {len(sampled_visitors)} visitors, {len(sample_items)} items")


if __name__ == "__main__":
    main()

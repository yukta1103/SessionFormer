import argparse
import json
from pathlib import Path

import numpy as np

from sessionformer.content.embeddings import build_item_content_embeddings
from sessionformer.data.vocab import ItemVocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vocab = ItemVocab.load(data_dir / "vocab.json")
    raw_item_text = json.loads((data_dir / "item_text.json").read_text())
    item_text = {int(k): v for k, v in raw_item_text.items()}

    embeddings = build_item_content_embeddings(item_text, vocab, model_name=args.model_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_content_embeddings.npy", embeddings)

    covered = sum(1 for idx in range(len(vocab)) if vocab.decode(idx) in item_text)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Vocab items with property text: {covered}/{len(vocab)}")


if __name__ == "__main__":
    main()

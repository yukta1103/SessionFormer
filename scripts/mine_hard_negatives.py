import argparse
import json
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.models.sasrec import SASRec
from sessionformer.negatives.hard_negative_mining import mine_hard_negatives
from sessionformer.utils.training import sessions_by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--sasrec-config", default="config/sasrec.yaml")
    parser.add_argument("--sasrec-checkpoint", default="checkpoints/sasrec_best.pt")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-hard-negatives", type=int, default=5)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sasrec_cfg = yaml.safe_load(Path(args.sasrec_config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_split, vocab = sessions_by_split(data_dir)

    model = SASRec(
        vocab_size=len(vocab),
        max_seq_len=sasrec_cfg["max_seq_len"],
        embedding_dim=sasrec_cfg["embedding_dim"],
        num_heads=sasrec_cfg["num_heads"],
        num_blocks=sasrec_cfg["num_blocks"],
        ff_hidden_dim=sasrec_cfg["ff_hidden_dim"],
        dropout=sasrec_cfg["dropout"],
    ).to(device)
    checkpoint = torch.load(args.sasrec_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        ds = SessionDataset(by_split[split], max_seq_len=sasrec_cfg["max_seq_len"])
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        hard_negatives = mine_hard_negatives(
            model, loader, device, top_k=args.top_k, num_hard_negatives=args.num_hard_negatives
        )
        (out_dir / f"hard_negatives_{split}.json").write_text(json.dumps(hard_negatives))

        num_examples = sum(len(negs) for negs in hard_negatives)
        empty = sum(1 for negs in hard_negatives if len(negs) == 0)
        print(f"[{split}] sessions: {len(hard_negatives)}, hard-negative examples: {num_examples}, sessions with none: {empty}")


if __name__ == "__main__":
    main()

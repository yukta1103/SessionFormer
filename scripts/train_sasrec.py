import argparse
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: loading torch's
# native libraries first causes an access violation when pyarrow initializes
# on this machine. See sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.models.sasrec import SASRec
from sessionformer.negatives.sampler import UniformNegativeSampler
from sessionformer.utils.seed import set_seed
from sessionformer.utils.training import sessions_by_split, train_with_early_stopping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sasrec.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/sasrec_best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    by_split, vocab = sessions_by_split(Path(args.data_dir))
    train_ds = SessionDataset(by_split["train"], max_seq_len=cfg["max_seq_len"])
    val_ds = SessionDataset(by_split["val"], max_seq_len=cfg["max_seq_len"])
    print(f"Train sessions: {len(train_ds)}, val sessions: {len(val_ds)}, vocab size: {len(vocab)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = SASRec(
        vocab_size=len(vocab),
        max_seq_len=cfg["max_seq_len"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        ff_hidden_dim=cfg["ff_hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    sampler = UniformNegativeSampler(vocab_size=len(vocab), num_negatives=cfg["num_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    train_with_early_stopping(
        model,
        sampler,
        optimizer,
        train_loader,
        val_loader,
        device,
        epochs=cfg["epochs"],
        patience=cfg.get("early_stopping_patience", 3),
        checkpoint_path=Path(args.checkpoint),
        config=cfg,
    )


if __name__ == "__main__":
    main()

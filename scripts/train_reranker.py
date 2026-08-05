import argparse
import json
import time
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.data.reranker_dataset import RerankerDataset
from sessionformer.models.reranker import Reranker
from sessionformer.utils.seed import set_seed
from sessionformer.utils.training import sessions_by_split


def _run_epoch(model, loader, device, optimizer=None):
    """Listwise training: each session's candidates are [target, *hard_negatives],
    trained with softmax cross-entropy over the group (target is always index 0).
    This directly optimizes relative ranking within a pool, unlike independent
    pointwise binary decisions, which don't guarantee correct ordering when many
    candidates are compared together at inference time."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for input_seq, lengths, candidates in loader:
            input_seq, lengths, candidates = input_seq.to(device), lengths.to(device), candidates.to(device)
            labels = torch.zeros(input_seq.size(0), dtype=torch.long, device=device)  # target is always index 0

            logits = model.score_candidates(input_seq, lengths, candidates)
            loss = F.cross_entropy(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_seq.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += input_seq.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reranker.yaml")
    parser.add_argument("--sasrec-config", default="config/sasrec.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/reranker_best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sasrec_cfg = yaml.safe_load(Path(args.sasrec_config).read_text())
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    by_split, vocab = sessions_by_split(data_dir)
    max_seq_len = sasrec_cfg["max_seq_len"]

    train_base = SessionDataset(by_split["train"], max_seq_len=max_seq_len)
    val_base = SessionDataset(by_split["val"], max_seq_len=max_seq_len)
    train_hard_negs = json.loads((data_dir / "hard_negatives_train.json").read_text())
    val_hard_negs = json.loads((data_dir / "hard_negatives_val.json").read_text())

    train_ds = RerankerDataset(train_base, train_hard_negs)
    val_ds = RerankerDataset(val_base, val_hard_negs)
    print(f"Train examples: {len(train_ds)}, val examples: {len(val_ds)}, vocab size: {len(vocab)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = Reranker(
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        ff_hidden_dim=cfg["ff_hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    patience = cfg.get("early_stopping_patience", 3)

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{cfg['epochs']} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
            print(f"  -> saved new best checkpoint ({checkpoint_path}), val_loss={val_loss:.4f}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No val_loss improvement for {patience} epochs, stopping early.")
                break

    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

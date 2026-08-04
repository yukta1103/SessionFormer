import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab
from sessionformer.models.gru4rec import GRU4Rec
from sessionformer.negatives.sampler import UniformNegativeSampler
from sessionformer.utils.seed import set_seed


def _sessions_by_split(processed_dir: Path):
    sessions = pd.read_parquet(processed_dir / "sessions.parquet")
    vocab = ItemVocab.load(processed_dir / "vocab.json")
    sessions["item_idx"] = sessions["itemid"].map(vocab.encode)

    by_split = {}
    for split in ("train", "val", "test"):
        subset = sessions[sessions["split"] == split]
        by_split[split] = list(subset.groupby("session_id")["item_idx"].apply(list))
    return by_split, vocab


def _train_step(model, sampler, input_seq, target_seq, device):
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    hidden = model.encode_sequence(input_seq)

    mask = (target_seq != PAD_IDX).float()
    pos_scores = model.score(hidden, target_seq.unsqueeze(-1)).squeeze(-1)
    negatives = sampler.sample(target_seq)
    neg_scores = model.score(hidden, negatives)

    pos_loss = -F.logsigmoid(pos_scores)
    neg_loss = -F.logsigmoid(-neg_scores).mean(dim=-1)
    per_position_loss = (pos_loss + neg_loss) * mask
    return per_position_loss.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate_recall_at_k(model, loader, device, k=10):
    model.eval()
    hits, total = 0, 0
    for input_seq, target_seq, lengths in loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        hidden = model.encode_sequence(input_seq)

        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = (lengths - 1).to(device)
        last_hidden = hidden[batch_idx, last_idx]
        target_last = target_seq[batch_idx, last_idx]

        scores = model.score_all_items(last_hidden)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")
        topk = scores.topk(k, dim=-1).indices

        hits += (topk == target_last.unsqueeze(-1)).any(dim=-1).sum().item()
        total += input_seq.size(0)
    model.train()
    return hits / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gru4rec.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/gru4rec_best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    by_split, vocab = _sessions_by_split(Path(args.data_dir))
    train_ds = SessionDataset(by_split["train"], max_seq_len=cfg["max_seq_len"])
    val_ds = SessionDataset(by_split["val"], max_seq_len=cfg["max_seq_len"])
    print(f"Train sessions: {len(train_ds)}, val sessions: {len(val_ds)}, vocab size: {len(vocab)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = GRU4Rec(
        vocab_size=len(vocab),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    sampler = UniformNegativeSampler(vocab_size=len(vocab), num_negatives=cfg["num_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        total_loss, num_batches = 0.0, 0
        for input_seq, target_seq, _ in train_loader:
            loss = _train_step(model, sampler, input_seq, target_seq, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        val_recall = evaluate_recall_at_k(model, val_loader, device, k=10)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{cfg['epochs']} | train_loss={total_loss / num_batches:.4f} "
            f"| val_recall@10={val_recall:.4f} | {elapsed:.1f}s"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
            print(f"  -> saved new best checkpoint ({checkpoint_path}), val_recall@10={val_recall:.4f}")

    print(f"Best val_recall@10: {best_recall:.4f}")


if __name__ == "__main__":
    main()

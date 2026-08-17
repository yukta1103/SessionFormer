# NOTE: this module imports pandas before torch on purpose. On this machine,
# loading torch's native libraries before pandas/pyarrow initializes causes an
# intermittent access-violation crash. Callers should likewise import pandas
# before torch at their own top level, since Python only runs a module's
# imports once (whichever import statement executes first in the process
# determines load order).
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F

from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab


def sessions_by_split(processed_dir: Path):
    sessions = pd.read_parquet(processed_dir / "sessions.parquet")
    vocab = ItemVocab.load(processed_dir / "vocab.json")
    sessions["item_idx"] = sessions["itemid"].map(vocab.encode)

    by_split = {}
    for split in ("train", "val", "test"):
        subset = sessions[sessions["split"] == split]
        by_split[split] = list(subset.groupby("session_id")["item_idx"].apply(list))
    return by_split, vocab


def train_step(model, sampler, input_seq: torch.Tensor, target_seq: torch.Tensor, device) -> torch.Tensor:
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    hidden = model.encode_sequence(input_seq)

    mask = (target_seq != PAD_IDX).float()
    pos_scores = model.score(hidden, target_seq.unsqueeze(-1)).squeeze(-1)
    negatives = sampler.sample(target_seq)

    # Explicit hard negative: the item already at this position (input_seq[i]
    # is exactly "the item immediately before the thing being predicted").
    # Uniform random sampling over a ~129k-item vocab almost never happens to
    # draw this specific item, so without folding it in here the loss never
    # directly penalizes echoing the input back out when that's wrong. Folded
    # into the same averaged negative pool (not a separate full-weight loss
    # term) so it gets proportional weight rather than doubling the
    # "push everything down" pressure relative to the positive signal.
    # Wherever the echo is genuinely correct (a real repeat click), it's
    # replaced with a harmless duplicate of an existing negative instead of
    # penalizing it.
    is_wrong_echo = input_seq != target_seq
    self_negative = torch.where(is_wrong_echo, input_seq, negatives[..., 0])
    all_negatives = torch.cat([negatives, self_negative.unsqueeze(-1)], dim=-1)
    neg_scores = model.score(hidden, all_negatives)

    pos_loss = -F.logsigmoid(pos_scores)
    neg_loss = -F.logsigmoid(-neg_scores).mean(dim=-1)
    per_position_loss = (pos_loss + neg_loss) * mask
    return per_position_loss.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate_recall_at_k(model, loader, device, k: int = 10) -> float:
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


def train_with_early_stopping(
    model,
    sampler,
    optimizer,
    train_loader,
    val_loader,
    device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
    config: Optional[dict] = None,
) -> float:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss, num_batches = 0.0, 0
        for input_seq, target_seq, _ in train_loader:
            loss = train_step(model, sampler, input_seq, target_seq, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        val_recall = evaluate_recall_at_k(model, val_loader, device, k=10)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{epochs} | train_loss={total_loss / num_batches:.4f} "
            f"| val_recall@10={val_recall:.4f} | {elapsed:.1f}s"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            epochs_since_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
            print(f"  -> saved new best checkpoint ({checkpoint_path}), val_recall@10={val_recall:.4f}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No val_recall@10 improvement for {patience} epochs, stopping early.")
                break

    print(f"Best val_recall@10: {best_recall:.4f}")
    return best_recall

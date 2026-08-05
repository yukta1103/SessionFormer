import argparse
import json
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sessionformer.content.fusion import cosine_scores, fuse_scores, session_content_vector
from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX, UNK_IDX
from sessionformer.models.sasrec import SASRec
from sessionformer.utils.training import sessions_by_split


@torch.no_grad()
def evaluate(model, item_content_embeddings, is_cold, loader, device, alpha, k=10):
    model.eval()
    stats = {
        "cold": {"hits_fused": 0, "hits_plain": 0, "total": 0},
        "warm": {"hits_fused": 0, "hits_plain": 0, "total": 0},
    }
    for input_seq, target_seq, lengths in loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        hidden = model.encode_sequence(input_seq)

        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = (lengths - 1).to(device)
        last_hidden = hidden[batch_idx, last_idx]
        target_last = target_seq[batch_idx, last_idx]

        collab_scores = model.score_all_items(last_hidden)

        query = session_content_vector(item_content_embeddings, input_seq)
        content_scores = cosine_scores(query, item_content_embeddings)
        fused_scores = fuse_scores(collab_scores, content_scores, is_cold, alpha)

        # Exclude PAD/UNK from ranking only now, after normalization/fusion —
        # doing this beforehand would feed -inf into the min-max
        # normalization and corrupt every score in the row.
        for scores in (collab_scores, fused_scores):
            scores[:, PAD_IDX] = float("-inf")
            scores[:, UNK_IDX] = float("-inf")

        plain_topk = collab_scores.topk(k, dim=-1).indices
        fused_topk = fused_scores.topk(k, dim=-1).indices

        hit_plain = (plain_topk == target_last.unsqueeze(-1)).any(dim=-1)
        hit_fused = (fused_topk == target_last.unsqueeze(-1)).any(dim=-1)
        target_is_cold = is_cold[target_last]

        for mask, bucket in [(target_is_cold, "cold"), (~target_is_cold, "warm")]:
            stats[bucket]["hits_plain"] += hit_plain[mask].sum().item()
            stats[bucket]["hits_fused"] += hit_fused[mask].sum().item()
            stats[bucket]["total"] += mask.sum().item()

    model.train()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--sasrec-config", default="config/sasrec.yaml")
    parser.add_argument("--sasrec-checkpoint", default="checkpoints/sasrec_best.pt")
    parser.add_argument("--fusion-config", default="config/content_fusion.yaml")
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sasrec_cfg = yaml.safe_load(Path(args.sasrec_config).read_text())
    fusion_cfg = yaml.safe_load(Path(args.fusion_config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_split, vocab = sessions_by_split(data_dir)
    val_ds = SessionDataset(by_split["val"], max_seq_len=sasrec_cfg["max_seq_len"])
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

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

    item_content_embeddings = torch.from_numpy(np.load(data_dir / "item_content_embeddings.npy")).to(device)

    cold_start_items = set(json.loads((data_dir / "cold_start_items.json").read_text()))
    is_cold = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    for item_id in cold_start_items:
        is_cold[vocab.encode(item_id)] = True

    alpha = args.alpha if args.alpha is not None else fusion_cfg["alpha"]
    stats = evaluate(model, item_content_embeddings, is_cold, val_loader, device, alpha, k=10)

    for bucket in ("cold", "warm"):
        s = stats[bucket]
        if s["total"] == 0:
            print(f"{bucket}: no val sessions with a {bucket} target")
            continue
        recall_plain = s["hits_plain"] / s["total"]
        recall_fused = s["hits_fused"] / s["total"]
        print(
            f"{bucket}-target sessions (n={s['total']}): "
            f"Recall@10 plain={recall_plain:.4f} | fused(alpha={alpha})={recall_fused:.4f}"
        )


if __name__ == "__main__":
    main()

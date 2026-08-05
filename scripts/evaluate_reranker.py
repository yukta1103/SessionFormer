import argparse
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX, UNK_IDX
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec
from sessionformer.utils.training import sessions_by_split


@torch.no_grad()
def evaluate(sasrec, reranker, loader, device, candidate_pool_k=20, k=10):
    """Returns per-example (entropy, hit_plain, hit_reranked) so callers can
    stratify by SASRec's own confidence -- the reranker is only meant to be
    invoked when SASRec is uncertain (Phase 7), so an average over all
    examples (mostly ones SASRec already gets right) can hide whether it
    actually helps in the regime it's meant for."""
    sasrec.eval()
    reranker.eval()
    entropies, hits_plain, hits_reranked = [], [], []

    for input_seq, target_seq, lengths in loader:
        input_seq, target_seq, lengths = input_seq.to(device), target_seq.to(device), lengths.to(device)
        hidden = sasrec.encode_sequence(input_seq)

        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = lengths - 1
        last_hidden = hidden[batch_idx, last_idx]
        target_last = target_seq[batch_idx, last_idx]

        scores = sasrec.score_all_items(last_hidden)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")

        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
        entropies.append(entropy.cpu())

        pool = scores.topk(candidate_pool_k, dim=-1).indices  # (B, candidate_pool_k)
        plain_topk = pool[:, :k]
        hits_plain.append((plain_topk == target_last.unsqueeze(-1)).any(dim=-1).cpu())

        rerank_logits = reranker.score_candidates(input_seq, lengths, pool)  # (B, candidate_pool_k)
        reranked_order = rerank_logits.argsort(dim=-1, descending=True)
        reranked_pool = torch.gather(pool, 1, reranked_order)
        reranked_topk = reranked_pool[:, :k]
        hits_reranked.append((reranked_topk == target_last.unsqueeze(-1)).any(dim=-1).cpu())

    return torch.cat(entropies), torch.cat(hits_plain), torch.cat(hits_reranked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--sasrec-config", default="config/sasrec.yaml")
    parser.add_argument("--sasrec-checkpoint", default="checkpoints/sasrec_best.pt")
    parser.add_argument("--reranker-config", default="config/reranker.yaml")
    parser.add_argument("--reranker-checkpoint", default="checkpoints/reranker_best.pt")
    parser.add_argument("--candidate-pool-k", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sasrec_cfg = yaml.safe_load(Path(args.sasrec_config).read_text())
    reranker_cfg = yaml.safe_load(Path(args.reranker_config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_split, vocab = sessions_by_split(data_dir)
    max_seq_len = sasrec_cfg["max_seq_len"]
    val_ds = SessionDataset(by_split["val"], max_seq_len=max_seq_len)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    sasrec = SASRec(
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=sasrec_cfg["embedding_dim"],
        num_heads=sasrec_cfg["num_heads"],
        num_blocks=sasrec_cfg["num_blocks"],
        ff_hidden_dim=sasrec_cfg["ff_hidden_dim"],
        dropout=sasrec_cfg["dropout"],
    ).to(device)
    sasrec.load_state_dict(torch.load(args.sasrec_checkpoint, map_location=device)["model_state"])

    reranker = Reranker(
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=reranker_cfg["embedding_dim"],
        num_heads=reranker_cfg["num_heads"],
        num_blocks=reranker_cfg["num_blocks"],
        ff_hidden_dim=reranker_cfg["ff_hidden_dim"],
        dropout=reranker_cfg["dropout"],
    ).to(device)
    reranker.load_state_dict(torch.load(args.reranker_checkpoint, map_location=device)["model_state"])

    entropy, hits_plain, hits_reranked = evaluate(sasrec, reranker, val_loader, device, args.candidate_pool_k, k=10)
    print(f"Plain SASRec top-10 Recall@10 (overall): {hits_plain.float().mean().item():.4f}")
    print(f"Reranked (pool={args.candidate_pool_k}) Recall@10 (overall): {hits_reranked.float().mean().item():.4f}")

    median = entropy.median()
    for label, mask in [("low-entropy half (confident)", entropy <= median), ("high-entropy half (uncertain)", entropy > median)]:
        n = mask.sum().item()
        print(
            f"{label} (n={n}): plain={hits_plain[mask].float().mean().item():.4f} "
            f"reranked={hits_reranked[mask].float().mean().item():.4f}"
        )


if __name__ == "__main__":
    main()

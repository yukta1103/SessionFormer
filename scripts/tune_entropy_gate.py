import argparse
import time
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas  # noqa: F401
import torch
import yaml
from torch.utils.data import DataLoader

from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX, UNK_IDX
from sessionformer.gating.entropy_gate import EntropyGate, softmax_entropy
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec
from sessionformer.utils.training import sessions_by_split


@torch.no_grad()
def collect_predictions(sasrec, reranker, loader, device, candidate_pool_k=20, k=10):
    """Per val example: entropy of SASRec's own distribution, whether the
    plain top-k hits the target, and whether the reranked top-k hits it."""
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
        entropies.append(softmax_entropy(scores).cpu())

        pool = scores.topk(candidate_pool_k, dim=-1).indices
        hits_plain.append((pool[:, :k] == target_last.unsqueeze(-1)).any(dim=-1).cpu())

        rerank_scores = reranker.score_candidates(input_seq, lengths, pool)
        reranked_order = rerank_scores.argsort(dim=-1, descending=True)
        reranked_pool = torch.gather(pool, 1, reranked_order)
        hits_reranked.append((reranked_pool[:, :k] == target_last.unsqueeze(-1)).any(dim=-1).cpu())

    return torch.cat(entropies), torch.cat(hits_plain), torch.cat(hits_reranked)


@torch.no_grad()
def measure_latency_ms(sasrec, reranker, loader, device, candidate_pool_k=20, num_batches=20):
    sasrec.eval()
    reranker.eval()
    sasrec_total, reranker_total, examples = 0.0, 0.0, 0

    for i, (input_seq, target_seq, lengths) in enumerate(loader):
        if i >= num_batches:
            break
        input_seq, lengths = input_seq.to(device), lengths.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = sasrec.encode_sequence(input_seq)
        last_idx = lengths - 1
        last_hidden = hidden[torch.arange(input_seq.size(0), device=device), last_idx]
        scores = sasrec.score_all_items(last_hidden)
        pool = scores.topk(candidate_pool_k, dim=-1).indices
        if device.type == "cuda":
            torch.cuda.synchronize()
        sasrec_total += time.perf_counter() - start

        start = time.perf_counter()
        reranker.score_candidates(input_seq, lengths, pool)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reranker_total += time.perf_counter() - start

        examples += input_seq.size(0)

    return (sasrec_total / examples) * 1000, (reranker_total / examples) * 1000


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
    test_ds = SessionDataset(by_split["test"], max_seq_len=max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

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

    entropy, hits_plain, hits_reranked = collect_predictions(sasrec, reranker, val_loader, device, args.candidate_pool_k)

    never_recall = hits_plain.float().mean().item()
    always_recall = hits_reranked.float().mean().item()
    print(f"Never-rerank  (0% fire):   Recall@10={never_recall:.4f}")
    print(f"Always-rerank (100% fire): Recall@10={always_recall:.4f}")
    print()

    percentiles = torch.linspace(0, 100, 21)
    thresholds = torch.quantile(entropy, percentiles / 100)

    best = None
    print(f"{'percentile':>10} {'threshold':>10} {'fire_rate':>10} {'recall':>8}")
    for pct, threshold in zip(percentiles.tolist(), thresholds.tolist()):
        gate = EntropyGate(threshold=threshold)
        fire = gate.should_refine(entropy)
        gated_hits = torch.where(fire, hits_reranked, hits_plain)
        recall = gated_hits.float().mean().item()
        fire_rate = fire.float().mean().item()
        print(f"{pct:>9.0f}% {threshold:>10.4f} {fire_rate:>9.1%} {recall:>8.4f}")

        # prefer higher recall; break ties toward a lower firing rate (cheaper)
        key = (recall, -fire_rate)
        if best is None or key > best[0]:
            best = (key, pct, threshold, fire_rate, recall)

    _, best_pct, best_threshold, best_fire_rate, best_recall = best
    print()
    print(f"Best on val: percentile={best_pct:.0f}% threshold={best_threshold:.4f} fire_rate={best_fire_rate:.1%} recall={best_recall:.4f}")

    # Honesty check: apply the val-tuned threshold to the untouched test split.
    test_entropy, test_hits_plain, test_hits_reranked = collect_predictions(
        sasrec, reranker, test_loader, device, args.candidate_pool_k
    )
    test_gate = EntropyGate(threshold=best_threshold)
    test_fire = test_gate.should_refine(test_entropy)
    test_gated_hits = torch.where(test_fire, test_hits_reranked, test_hits_plain)
    print()
    print(
        f"Test (held-out, threshold fixed from val): never-rerank={test_hits_plain.float().mean().item():.4f} "
        f"always-rerank={test_hits_reranked.float().mean().item():.4f} "
        f"gated(fire_rate={test_fire.float().mean().item():.1%})={test_gated_hits.float().mean().item():.4f}"
    )

    sasrec_ms, reranker_ms = measure_latency_ms(sasrec, reranker, val_loader, device, args.candidate_pool_k)
    print()
    print(f"Per-example latency: SASRec scoring={sasrec_ms:.3f}ms, reranker (pool={args.candidate_pool_k})={reranker_ms:.3f}ms")
    always_extra_ms = reranker_ms
    gated_extra_ms = best_fire_rate * reranker_ms
    savings_pct = (1 - gated_extra_ms / always_extra_ms) * 100 if always_extra_ms > 0 else 0.0
    print(
        f"Extra latency vs SASRec-only: always-rerank=+{always_extra_ms:.3f}ms/example, "
        f"gated (fire_rate={best_fire_rate:.1%})=+{gated_extra_ms:.3f}ms/example "
        f"({savings_pct:.1f}% less reranking compute than always-rerank)"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import time
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from eval.metrics import mrr, ndcg_at_k, recall_at_k
from sessionformer.baselines.item_knn import ItemKNNBaseline
from sessionformer.baselines.popularity import PopularityBaseline
from sessionformer.data.dataset import SessionDataset
from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab
from sessionformer.gating.entropy_gate import EntropyGate, softmax_entropy
from sessionformer.models.gru4rec import GRU4Rec
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec
from sessionformer.utils.training import sessions_by_split

TOPK = 20
NO_PRED = -1
STRATA = ["overall", "warm_item", "cold_item", "warm_user", "cold_user"]


def _pad_predictions(preds: list, width: int = TOPK) -> np.ndarray:
    arr = np.full((len(preds), width), NO_PRED, dtype=np.int64)
    for i, row in enumerate(preds):
        arr[i, : min(len(row), width)] = row[:width]
    return arr


def predict_baseline(model, sessions: list) -> tuple:
    preds = []
    start = time.perf_counter()
    for items in sessions:
        preds.append(model.recommend(items[:-1], k=TOPK))
    elapsed_ms = (time.perf_counter() - start) / len(sessions) * 1000
    return _pad_predictions(preds), elapsed_ms


@torch.no_grad()
def predict_neural(model, loader, device) -> tuple:
    model.eval()
    all_topk = []
    total_time, total_examples = 0.0, 0
    for input_seq, target_seq, lengths in loader:
        input_seq, lengths = input_seq.to(device), lengths.to(device)
        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = lengths - 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = model.encode_sequence(input_seq)
        last_hidden = hidden[batch_idx, last_idx]
        scores = model.score_all_items(last_hidden)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")
        topk = scores.topk(TOPK, dim=-1).indices
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_examples += input_seq.size(0)

        all_topk.append(topk.cpu().numpy())

    model.train()
    return np.concatenate(all_topk), (total_time / total_examples) * 1000


@torch.no_grad()
def predict_sasrec_family(sasrec, reranker, gate_threshold, loader, device) -> tuple:
    sasrec.eval()
    reranker.eval()
    all_pool, all_reranked = [], []
    entropies = []
    sasrec_time, rerank_time, total = 0.0, 0.0, 0

    for input_seq, target_seq, lengths in loader:
        input_seq, lengths = input_seq.to(device), lengths.to(device)
        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = lengths - 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = sasrec.encode_sequence(input_seq)
        last_hidden = hidden[batch_idx, last_idx]
        scores = sasrec.score_all_items(last_hidden)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")
        pool = scores.topk(TOPK, dim=-1).indices
        if device.type == "cuda":
            torch.cuda.synchronize()
        sasrec_time += time.perf_counter() - t0

        entropies.append(softmax_entropy(scores).cpu())

        t0 = time.perf_counter()
        rerank_scores = reranker.score_candidates(input_seq, lengths, pool)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rerank_time += time.perf_counter() - t0

        reranked_order = rerank_scores.argsort(dim=-1, descending=True)
        reranked_pool = torch.gather(pool, 1, reranked_order)

        all_pool.append(pool.cpu().numpy())
        all_reranked.append(reranked_pool.cpu().numpy())
        total += input_seq.size(0)

    sasrec.train()
    reranker.train()

    pool = np.concatenate(all_pool)
    reranked = np.concatenate(all_reranked)
    entropy = torch.cat(entropies)

    gate = EntropyGate(threshold=gate_threshold)
    fire = gate.should_refine(entropy).numpy()
    gated = np.where(fire[:, None], reranked, pool)

    sasrec_ms = sasrec_time / total * 1000
    rerank_ms = rerank_time / total * 1000
    fire_rate = float(fire.mean())

    predictions = {"SASRec": pool, "SASRec+AlwaysRerank": reranked, "SASRec+GatedRerank": gated}
    latencies = {
        "SASRec": sasrec_ms,
        "SASRec+AlwaysRerank": sasrec_ms + rerank_ms,
        "SASRec+GatedRerank": sasrec_ms + fire_rate * rerank_ms,
    }
    return predictions, latencies, fire_rate


def compute_metrics(topk: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "n": len(targets),
        "Recall@10": recall_at_k(topk, targets, 10),
        "Recall@20": recall_at_k(topk, targets, 20),
        "NDCG@10": ndcg_at_k(topk, targets, 10),
        "NDCG@20": ndcg_at_k(topk, targets, 20),
        "MRR": mrr(topk, targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--gru4rec-config", default="config/gru4rec.yaml")
    parser.add_argument("--gru4rec-checkpoint", default="checkpoints/gru4rec_best.pt")
    parser.add_argument("--sasrec-config", default="config/sasrec.yaml")
    parser.add_argument("--sasrec-checkpoint", default="checkpoints/sasrec_best.pt")
    parser.add_argument("--reranker-config", default="config/reranker.yaml")
    parser.add_argument("--reranker-checkpoint", default="checkpoints/reranker_best.pt")
    parser.add_argument("--gate-config", default="checkpoints/entropy_gate_threshold.json")
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = ItemVocab.load(data_dir / "vocab.json")
    cold_items_raw = set(json.loads((data_dir / "cold_start_items.json").read_text()))
    cold_users_raw = set(json.loads((data_dir / "cold_start_users.json").read_text()))
    is_cold_item = {vocab.encode(item_id) for item_id in cold_items_raw}

    by_split, _ = sessions_by_split(data_dir)

    sessions_df = pd.read_parquet(data_dir / "sessions.parquet")
    test_visitors = sessions_df[sessions_df["split"] == "test"].groupby("session_id")["visitorid"].first().tolist()

    max_seq_len = yaml.safe_load(Path(args.sasrec_config).read_text())["max_seq_len"]
    test_ds = SessionDataset(by_split["test"], max_seq_len=max_seq_len)
    if len(test_ds) != len(test_visitors):
        raise RuntimeError("test session count mismatch between SessionDataset and visitorid lookup")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    targets = np.array([s[-1] for s in test_ds.sessions])
    target_is_cold_item = np.array([t in is_cold_item for t in targets])
    target_is_cold_user = np.array([v in cold_users_raw for v in test_visitors])

    predictions: dict = {}
    latencies: dict = {}

    # --- Popularity & Item-KNN ---
    train_items = [item for session in by_split["train"] for item in session]
    popularity = PopularityBaseline().fit(train_items)
    item_knn = ItemKNNBaseline().fit(by_split["train"], vocab_size=len(vocab))
    predictions["Popularity"], latencies["Popularity"] = predict_baseline(popularity, test_ds.sessions)
    predictions["ItemKNN"], latencies["ItemKNN"] = predict_baseline(item_knn, test_ds.sessions)

    # --- GRU4Rec ---
    gru_cfg = yaml.safe_load(Path(args.gru4rec_config).read_text())
    gru4rec = GRU4Rec(
        vocab_size=len(vocab),
        embedding_dim=gru_cfg["embedding_dim"],
        hidden_dim=gru_cfg["hidden_dim"],
        num_layers=gru_cfg["num_layers"],
    ).to(device)
    gru4rec.load_state_dict(torch.load(args.gru4rec_checkpoint, map_location=device)["model_state"])
    predictions["GRU4Rec"], latencies["GRU4Rec"] = predict_neural(gru4rec, test_loader, device)

    # --- SASRec family ---
    sasrec_cfg = yaml.safe_load(Path(args.sasrec_config).read_text())
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

    reranker_cfg = yaml.safe_load(Path(args.reranker_config).read_text())
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

    gate_threshold = json.loads(Path(args.gate_config).read_text())["threshold"]
    sasrec_preds, sasrec_latencies, fire_rate = predict_sasrec_family(
        sasrec, reranker, gate_threshold, test_loader, device
    )
    predictions.update(sasrec_preds)
    latencies.update(sasrec_latencies)

    # --- Metrics table ---
    rows = []
    for method, topk in predictions.items():
        strata_masks = {
            "overall": np.ones(len(targets), dtype=bool),
            "warm_item": ~target_is_cold_item,
            "cold_item": target_is_cold_item,
            "warm_user": ~target_is_cold_user,
            "cold_user": target_is_cold_user,
        }
        for stratum in STRATA:
            mask = strata_masks[stratum]
            if mask.sum() == 0:
                continue
            row = {"method": method, "stratum": stratum, "latency_ms": latencies[method]}
            row.update(compute_metrics(topk[mask], targets[mask]))
            rows.append(row)

    table = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    overall = table[table["stratum"] == "overall"].drop(columns=["stratum"])
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(
        f"Note: reranking only reorders SASRec's own top-{TOPK} pool without expanding it, so "
        f"Recall@20 is identical across SASRec/AlwaysRerank/GatedRerank by construction; the "
        f"deltas show up in Recall@10, NDCG, and MRR where ordering matters."
    )
    print(f"Entropy gate fire rate on test: {fire_rate:.1%}")
    print()
    print("Cold-start breakdown (item):")
    cold_item_table = table[table["stratum"].isin(["warm_item", "cold_item"])].drop(columns=["latency_ms"])
    print(cold_item_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Cold-start breakdown (user):")
    cold_user_table = table[table["stratum"].isin(["warm_user", "cold_user"])].drop(columns=["latency_ms"])
    print(cold_user_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Full results saved to {out_dir / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()

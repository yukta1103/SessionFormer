import json
import random
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note.
import pandas as pd
import streamlit as st
import torch
import yaml

from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab
from sessionformer.gating.entropy_gate import EntropyGate, softmax_entropy
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec

DATA_DIR = Path("data/processed")
MIN_LEN, MAX_LEN = 4, 15
TOP_K = 10


@st.cache_resource
def load_artifacts():
    sasrec_cfg = yaml.safe_load(Path("config/sasrec.yaml").read_text())
    reranker_cfg = yaml.safe_load(Path("config/reranker.yaml").read_text())
    gate_cfg = json.loads(Path("checkpoints/entropy_gate_threshold.json").read_text())

    vocab = ItemVocab.load(DATA_DIR / "vocab.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = sasrec_cfg["max_seq_len"]

    sasrec = SASRec(
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=sasrec_cfg["embedding_dim"],
        num_heads=sasrec_cfg["num_heads"],
        num_blocks=sasrec_cfg["num_blocks"],
        ff_hidden_dim=sasrec_cfg["ff_hidden_dim"],
        dropout=sasrec_cfg["dropout"],
    ).to(device)
    sasrec.load_state_dict(torch.load("checkpoints/sasrec_best.pt", map_location=device)["model_state"])
    sasrec.eval()

    reranker = Reranker(
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=reranker_cfg["embedding_dim"],
        num_heads=reranker_cfg["num_heads"],
        num_blocks=reranker_cfg["num_blocks"],
        ff_hidden_dim=reranker_cfg["ff_hidden_dim"],
        dropout=reranker_cfg["dropout"],
    ).to(device)
    reranker.load_state_dict(torch.load("checkpoints/reranker_best.pt", map_location=device)["model_state"])
    reranker.eval()

    gate = EntropyGate(threshold=gate_cfg["threshold"])
    cold_items = set(json.loads((DATA_DIR / "cold_start_items.json").read_text()))

    return sasrec, reranker, gate, vocab, device, max_seq_len, cold_items


@st.cache_data
def load_demo_sessions():
    sessions = pd.read_parquet(DATA_DIR / "sessions.parquet")
    test = sessions[sessions["split"] == "test"]
    grouped = test.groupby("session_id")["itemid"].apply(list)
    return [items for items in grouped if MIN_LEN <= len(items) <= MAX_LEN]


def get_predictions(sasrec, reranker, gate, vocab, device, max_seq_len, item_ids):
    item_indices = [vocab.encode(i) for i in item_ids]
    input_items = item_indices[:-1]
    length = len(input_items)

    padded = input_items + [PAD_IDX] * (max_seq_len - length)
    input_seq = torch.tensor([padded], dtype=torch.long, device=device)

    with torch.no_grad():
        hidden = sasrec.encode_sequence(input_seq)
        scores = sasrec.score_all_items(hidden[0])
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")
        entropy = softmax_entropy(scores)
        topk = scores.topk(TOP_K, dim=-1).indices

    steps = []
    for pos in range(length):
        step_entropy = entropy[pos].item()
        fires = bool(gate.should_refine(entropy[pos]).item())
        plain_topk_ids = [vocab.decode(i) for i in topk[pos].tolist()]

        reranked_topk_ids = None
        if fires:
            with torch.no_grad():
                pool = topk[pos : pos + 1]
                lengths = torch.tensor([pos + 1], device=device)
                rerank_scores = reranker.score_candidates(input_seq, lengths, pool)
                order = rerank_scores.argsort(dim=-1, descending=True)
                reranked_pool = torch.gather(pool, 1, order)[0].tolist()
            reranked_topk_ids = [vocab.decode(i) for i in reranked_pool]

        steps.append(
            {
                "context_ids": item_ids[: pos + 1],
                "target_id": item_ids[pos + 1],
                "entropy": step_entropy,
                "gate_fired": fires,
                "plain_topk_ids": plain_topk_ids,
                "reranked_topk_ids": reranked_topk_ids,
            }
        )
    return steps


def clamp_step(step: int, num_steps: int) -> int:
    """Keeps a step index within [0, num_steps - 1], including the
    num_steps == 0 edge case (clamps to 0 rather than going negative)."""
    return max(0, min(step, num_steps - 1))


def main() -> None:
    st.set_page_config(page_title="SessionFormer demo", layout="wide")
    st.title("SessionFormer: session-by-session demo")
    st.caption(
        "Item IDs are RetailRocket's raw anonymized product IDs -- the dataset has no real names or images."
    )

    sasrec, reranker, gate, vocab, device, max_seq_len, cold_items = load_artifacts()
    demo_sessions = load_demo_sessions()

    if "session_items" not in st.session_state:
        st.session_state.session_items = random.choice(demo_sessions)
        st.session_state.step = 0

    if st.button("Shuffle session"):
        st.session_state.session_items = random.choice(demo_sessions)
        st.session_state.step = 0

    item_ids = st.session_state.session_items
    steps = get_predictions(sasrec, reranker, gate, vocab, device, max_seq_len, item_ids)
    num_steps = len(steps)
    st.session_state.step = clamp_step(st.session_state.step, num_steps)

    nav_col1, nav_col2, _ = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("Previous step", disabled=st.session_state.step == 0):
            st.session_state.step -= 1
    with nav_col2:
        if st.button("Next step", disabled=st.session_state.step >= num_steps - 1):
            st.session_state.step += 1

    # Re-clamp: the button handlers above mutate step after the clamp
    # above ran, so indexing must not trust that earlier clamp alone.
    st.session_state.step = clamp_step(st.session_state.step, num_steps)
    step = steps[st.session_state.step]
    st.subheader(f"Step {st.session_state.step + 1} of {num_steps}")

    st.write("**Session so far:** " + " -> ".join(f"#{i}" for i in step["context_ids"]))
    target_is_cold = step["target_id"] in cold_items
    cold_note = " _(cold-start item)_" if target_is_cold else ""
    st.write(f"**True next item:** #{step['target_id']}{cold_note}")

    entropy_col, gate_col = st.columns(2)
    with entropy_col:
        st.metric("Prediction entropy", f"{step['entropy']:.3f}")
        st.progress(min(step["entropy"] / (gate.threshold * 2), 1.0))
        st.caption(f"Gate threshold: {gate.threshold:.3f}")
    with gate_col:
        if step["gate_fired"]:
            st.success("Entropy gate FIRED — reranker invoked")
        else:
            st.info("Entropy gate did not fire — SASRec's top-10 trusted directly")

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        st.write("**SASRec top-10**")
        for rank, item_id in enumerate(step["plain_topk_ids"], start=1):
            marker = " <- true next item" if item_id == step["target_id"] else ""
            st.write(f"{rank}. #{item_id}{marker}")

    with pred_col2:
        if step["gate_fired"]:
            st.write("**Reranked top-10**")
            for rank, item_id in enumerate(step["reranked_topk_ids"], start=1):
                marker = " <- true next item" if item_id == step["target_id"] else ""
                st.write(f"{rank}. #{item_id}{marker}")
        else:
            st.write("_Reranker not invoked at this step_")


if __name__ == "__main__":
    main()

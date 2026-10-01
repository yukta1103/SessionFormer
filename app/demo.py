import json
import random
from pathlib import Path

# Import pandas (which eagerly loads pyarrow) before torch: see
# sessionformer/utils/training.py's docstring note. Still needed here even
# though the demo no longer reads sessions.parquet, because predictions_
# dataframe() below builds a pd.DataFrame for the bar charts -- the DLL
# ordering risk is about import order, not about which pandas API is used.
import pandas as pd
import streamlit as st
import torch
import yaml

from sessionformer.data.vocab import PAD_IDX, UNK_IDX, ItemVocab
from sessionformer.gating.entropy_gate import EntropyGate, softmax_entropy
from sessionformer.models.reranker import Reranker
from sessionformer.models.sasrec import SASRec

DATA_DIR = Path("data/processed")
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
    """Pre-filtered test-split sessions (length 4-15), exported once by
    scripts/export_demo_assets.py -- a small, self-contained artifact
    instead of bundling the full sessions.parquet (train+val+test, ~19MB)
    just to source a handful of demo sessions."""
    return json.loads((DATA_DIR / "demo_sessions.json").read_text())


@st.cache_data
def load_item_categories() -> dict:
    path = DATA_DIR / "demo_item_categories.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def format_item(item_id: int, categories: dict) -> str:
    category = categories.get(item_id)
    return f"#{item_id} (category {category})" if category else f"#{item_id}"


def clamp_step(step: int, num_steps: int) -> int:
    """Keeps a step index within [0, num_steps - 1], including the
    num_steps == 0 edge case (clamps to 0 rather than going negative)."""
    return max(0, min(step, num_steps - 1))


def score_step(is_hit: bool, step_index: int, scored_steps: set, hits: int, total: int):
    """Updates the running tally the first time a given step is viewed;
    revisiting a step via Previous/Next doesn't double-count it."""
    if step_index in scored_steps:
        return scored_steps, hits, total
    scored_steps = scored_steps | {step_index}
    total += 1
    if is_hit:
        hits += 1
    return scored_steps, hits, total


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
        probs = torch.softmax(scores, dim=-1)
        topk = scores.topk(TOP_K, dim=-1).indices

    steps = []
    for pos in range(length):
        step_entropy = entropy[pos].item()
        fires = bool(gate.should_refine(entropy[pos]).item())
        plain_topk_ids = [vocab.decode(i) for i in topk[pos].tolist()]
        plain_topk_probs = probs[pos, topk[pos]].tolist()

        reranked_topk_ids = None
        reranked_topk_scores = None
        if fires:
            with torch.no_grad():
                pool = topk[pos : pos + 1]
                lengths = torch.tensor([pos + 1], device=device)
                rerank_scores = reranker.score_candidates(input_seq, lengths, pool)
                order = rerank_scores.argsort(dim=-1, descending=True)
                reranked_pool = torch.gather(pool, 1, order)[0].tolist()
                reranked_scores_sorted = torch.gather(rerank_scores, 1, order)
                reranked_topk_scores = torch.softmax(reranked_scores_sorted, dim=-1)[0].tolist()
            reranked_topk_ids = [vocab.decode(i) for i in reranked_pool]

        target_id = item_ids[pos + 1]
        final_topk_ids = reranked_topk_ids if fires else plain_topk_ids

        steps.append(
            {
                "context_ids": item_ids[: pos + 1],
                "target_id": target_id,
                "entropy": step_entropy,
                "gate_fired": fires,
                "plain_topk_ids": plain_topk_ids,
                "plain_topk_probs": plain_topk_probs,
                "reranked_topk_ids": reranked_topk_ids,
                "reranked_topk_scores": reranked_topk_scores,
                "final_topk_ids": final_topk_ids,
                "is_hit": target_id in final_topk_ids,
            }
        )
    return steps


def predictions_dataframe(item_ids, values, target_id, categories: dict) -> pd.DataFrame:
    """Builds a chart-ready table: one row per candidate, its score, and
    whether it's the item that actually happened next -- used to color the
    correct bar differently from the rest."""
    rows = [
        {
            "item": format_item(item_id, categories),
            "score": value,
            "type": "Correct next item" if item_id == target_id else "Other guess",
        }
        for item_id, value in zip(item_ids, values)
    ]
    return pd.DataFrame(rows).set_index("item")


def _new_session(demo_sessions) -> None:
    st.session_state.session_items = random.choice(demo_sessions)
    st.session_state.step = 0
    st.session_state.scored_steps = set()
    st.session_state.hits = 0
    st.session_state.total = 0


def main() -> None:
    st.set_page_config(page_title="SessionFormer demo", layout="wide")
    st.title("SessionFormer: watch it predict, live")
    st.markdown(
        "A next-click predictor trained on real (anonymized) e-commerce browsing sessions. "
        "Step through a real session and watch it guess what the shopper clicked next — and "
        "whether it decided a slower, more careful second check was worth the extra time."
    )
    with st.expander("How to read this"):
        st.markdown(
            "- **Model confidence** reflects how sure the model is about its own top guess, based "
            "on patterns it learned during training — not whether that guess is actually correct.\n"
            "- **Entropy gate**: when confidence is low, a slower \"second opinion\" model (the "
            "reranker) is called in to double-check the top-10 order. When confidence is high, "
            "that extra step is skipped to save compute.\n"
            "- Item IDs are RetailRocket's raw anonymized product IDs — there are no real names or "
            "images in this dataset. A category code is shown alongside each ID where available."
        )

    sasrec, reranker, gate, vocab, device, max_seq_len, cold_items = load_artifacts()
    demo_sessions = load_demo_sessions()
    categories = load_item_categories()

    if "session_items" not in st.session_state:
        _new_session(demo_sessions)

    if st.button("Shuffle session"):
        _new_session(demo_sessions)

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

    is_new_step = st.session_state.step not in st.session_state.scored_steps
    st.session_state.scored_steps, st.session_state.hits, st.session_state.total = score_step(
        step["is_hit"], st.session_state.step, st.session_state.scored_steps, st.session_state.hits, st.session_state.total
    )
    if is_new_step and step["is_hit"]:
        st.balloons()

    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        st.subheader(f"Step {st.session_state.step + 1} of {num_steps}")
    with header_col2:
        st.metric("Score so far", f"{st.session_state.hits}/{st.session_state.total} correct")

    context_str = " -> ".join(format_item(i, categories) for i in step["context_ids"])
    st.write(f"**Session so far:** {context_str}")

    target_label = format_item(step["target_id"], categories)
    cold_note = " _(cold-start item — little training history)_" if step["target_id"] in cold_items else ""
    st.write(f"**What the shopper actually clicked next:** {target_label}{cold_note}")

    if step["is_hit"]:
        st.success(f"Correct — {target_label} was in the top 10")
    else:
        st.error(f"Missed — {target_label} was not in the top 10")

    conf_col, gate_col = st.columns(2)
    with conf_col:
        st.metric("Model confidence", "High" if not step["gate_fired"] else "Low")
        meter = min(step["entropy"] / (gate.threshold * 2), 1.0) if gate.threshold > 0 else 0.0
        st.progress(meter)
        st.caption(f"Raw uncertainty score: {step['entropy']:.3f} (gate threshold: {gate.threshold:.3f})")
    with gate_col:
        if step["gate_fired"]:
            st.info("Confidence was low, so the second-opinion model was called in to double-check.")
        else:
            st.info("Confidence was high enough that the fast model's guess was trusted directly.")

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        st.write("**Fast model's top 10** _(bar height = how strongly it believes each guess)_")
        plain_df = predictions_dataframe(step["plain_topk_ids"], step["plain_topk_probs"], step["target_id"], categories)
        st.bar_chart(plain_df, y="score", color="type", horizontal=True)

    with pred_col2:
        if step["gate_fired"]:
            st.write("**After the second opinion (reordered)**")
            reranked_df = predictions_dataframe(
                step["reranked_topk_ids"], step["reranked_topk_scores"], step["target_id"], categories
            )
            st.bar_chart(reranked_df, y="score", color="type", horizontal=True)
        else:
            st.write("_Second opinion skipped this time — the fast model was confident enough_")


if __name__ == "__main__":
    main()

# SessionFormer

A from-scratch transformer-based sequential recommender system with entropy-gated reranking, trained on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

SessionFormer predicts the next item a user will interact with given their session history. It implements SASRec (self-attentive sequential recommendation) from scratch, fuses collaborative and content signals for cold-start items, and uses an entropy-gated cross-encoder reranker — trained on hard negatives mined from the model's own predictions — to refine top-K predictions only when the model's own confidence is low.

## Techniques reused across this portfolio

- **Hard-negative-mined reranking** — same pattern as the hybrid-retrieval RAG project's hard-negative mining, applied here to rerank SASRec's top-K candidates.
- **Entropy-gated compute** — same pattern as the Quiet-STaR entropy-gating project, applied here to decide per-prediction-step whether the expensive reranker needs to run at all.

## Project status

Building phase by phase. See below for the plan; each phase is confirmed complete before starting the next.

1. Data pipeline (sessionization, vocab, time-based split, cold-start holdout)
2. Popularity + Item-KNN baselines
3. GRU4Rec baseline
4. SASRec (from scratch)
5. Cold-start content fusion (sentence-transformer embeddings)
6. Hard-negative mining + cross-encoder reranker
7. Entropy-gated reranking
8. Evaluation (Recall@10/20, NDCG@10/20, MRR; warm vs. cold-start)
9. Streamlit demo

## Repository structure

```
config/            YAML configs per phase
data/
  raw/             full RetailRocket CSVs (gitignored, user-downloaded)
  sample/          small checked-in subset — lets the pipeline run with no download
  processed/       sessionized data, vocab, splits (gitignored)
sessionformer/      core package
  data/            sessionization, vocab, splits, Dataset/DataLoader
  baselines/       popularity, item-KNN
  models/          gru4rec, sasrec, shared layers, reranker
  content/         sentence-transformer item embeddings, cold-start fusion
  negatives/       negative sampling, hard-negative mining
  gating/          entropy gate over a generic refiner interface
  utils/           config, seeding, checkpointing
scripts/            CLI entry points
notebooks/          exploration
eval/               metrics, comparison table, results (gitignored)
app/                Streamlit demo
tests/              unit tests
checkpoints/        model weights (gitignored)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

`torch` must be installed first, from PyTorch's own CUDA index — a plain `pip install -r requirements.txt` would otherwise pull a CPU-only wheel from PyPI's default index. Pick the CUDA tag (`cu121`/`cu124`/`cu128`/...) and torch version pairing that's available for your driver: check with `nvidia-smi` (look at the reported "CUDA Version", which is the max your driver supports) and `pip index versions torch --index-url https://download.pytorch.org/whl/<tag>`.

Verify GPU visibility:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset

Download from [Kaggle: RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) and place `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, and `category_tree.csv` into `data/raw/`. A small sample subset is checked into `data/sample/` so the pipeline can be run end-to-end without downloading the full dataset first.

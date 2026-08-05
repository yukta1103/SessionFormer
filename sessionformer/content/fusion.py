import torch
import torch.nn.functional as F

from sessionformer.data.vocab import PAD_IDX


def session_content_vector(item_embeddings: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
    """Mean content embedding of the (non-padding) items in each session.
    input_seq: (batch, seq_len) vocab indices -> (batch, embedding_dim)."""
    embedded = item_embeddings[input_seq]  # (batch, seq_len, embedding_dim)
    mask = (input_seq != PAD_IDX).float().unsqueeze(-1)
    summed = (embedded * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def cosine_scores(query: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
    """query: (batch, embedding_dim), item_embeddings: (vocab_size, embedding_dim)
    -> (batch, vocab_size) cosine similarity."""
    query_norm = F.normalize(query, dim=-1)
    items_norm = F.normalize(item_embeddings, dim=-1)
    return query_norm @ items_norm.T


def _row_minmax(scores: torch.Tensor) -> torch.Tensor:
    min_vals = scores.min(dim=-1, keepdim=True).values
    max_vals = scores.max(dim=-1, keepdim=True).values
    return (scores - min_vals) / (max_vals - min_vals).clamp(min=1e-8)


def fuse_scores(collab_scores: torch.Tensor, content_scores: torch.Tensor, is_cold: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blends collaborative and content scores for cold-start items only;
    warm items keep their (min-max normalized) collaborative score.
    collab_scores, content_scores: (batch, vocab_size). is_cold: (vocab_size,) bool."""
    collab_norm = _row_minmax(collab_scores)
    content_norm = _row_minmax(content_scores)
    fused = collab_norm.clone()
    fused[:, is_cold] = (1 - alpha) * collab_norm[:, is_cold] + alpha * content_norm[:, is_cold]
    return fused

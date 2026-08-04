import torch
import torch.nn as nn

from sessionformer.data.vocab import PAD_IDX


class GRU4Rec(nn.Module):
    """GRU-based sequential recommender with a tied item-embedding output
    (score = dot product of GRU hidden state and item embedding) so scoring
    against a small negative sample never requires a full vocab-sized output
    layer."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim) if hidden_dim != embedding_dim else nn.Identity()

    def encode_sequence(self, input_seq: torch.Tensor) -> torch.Tensor:
        """input_seq: (batch, seq_len) -> hidden states (batch, seq_len, embedding_dim)."""
        embedded = self.item_embedding(input_seq)
        hidden, _ = self.gru(embedded)
        return self.output_proj(hidden)

    def score(self, hidden: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """hidden: (batch, seq_len, embedding_dim)
        item_indices: (batch, seq_len, num_candidates) -> scores (batch, seq_len, num_candidates)."""
        item_vectors = self.item_embedding(item_indices)
        return (hidden.unsqueeze(-2) * item_vectors).sum(-1)

    def score_all_items(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (..., embedding_dim) -> scores (..., vocab_size), for evaluation only."""
        return hidden @ self.item_embedding.weight.T

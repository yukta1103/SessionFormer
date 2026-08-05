from typing import Optional

import torch
import torch.nn as nn

from sessionformer.data.vocab import PAD_IDX
from sessionformer.models.layers import SASRecBlock


class SASRec(nn.Module):
    """Self-attentive sequential recommender, built from scratch: item
    embedding + learned positional embedding, stacked causal self-attention
    blocks, and a tied-embedding scoring head (same interface as GRU4Rec:
    encode_sequence / score / score_all_items)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        ff_hidden_dim = ff_hidden_dim or embedding_dim * 4
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(embedding_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_blocks)
        )

    def encode_sequence(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_seq.shape
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch, seq_len)
        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x

    def score(self, hidden: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        item_vectors = self.item_embedding(item_indices)
        return (hidden.unsqueeze(-2) * item_vectors).sum(-1)

    def score_all_items(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_embedding.weight.T

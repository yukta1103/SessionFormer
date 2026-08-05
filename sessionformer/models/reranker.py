from typing import Optional

import torch
import torch.nn as nn

from sessionformer.data.vocab import PAD_IDX
from sessionformer.models.layers import SASRecBlock


class Reranker(nn.Module):
    """Lightweight cross-encoder: the candidate item is appended as an extra
    token right after the real context (at its true per-sample length, not
    the padded end), so causal self-attention lets the candidate's final
    hidden state be computed jointly with the whole context — unlike
    SASRec's separate-encode-then-dot-product scoring."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 32,
        num_heads: int = 2,
        num_blocks: int = 1,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        ff_hidden_dim = ff_hidden_dim or embedding_dim * 4
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.position_embedding = nn.Embedding(max_seq_len + 1, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(embedding_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_blocks)
        )
        self.score_head = nn.Linear(embedding_dim, 1)

    def build_input(self, input_seq: torch.Tensor, lengths: torch.Tensor, candidate_items: torch.Tensor):
        """input_seq: (batch, max_seq_len) right-padded context.
        lengths: (batch,) real context lengths. candidate_items: (batch,)
        Returns (extended_seq, candidate_pos), extended_seq: (batch, max_seq_len + 1)
        with the candidate written at each sample's true context length."""
        batch_size = input_seq.size(0)
        extended = torch.zeros(batch_size, self.max_seq_len + 1, dtype=torch.long, device=input_seq.device)
        extended[:, : self.max_seq_len] = input_seq
        extended[torch.arange(batch_size, device=input_seq.device), lengths] = candidate_items
        return extended, lengths

    def forward(self, input_seq: torch.Tensor, lengths: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        extended, candidate_pos = self.build_input(input_seq, lengths, candidate_items)
        batch, seq_len = extended.shape

        positions = torch.arange(seq_len, device=extended.device).unsqueeze(0).expand(batch, seq_len)
        x = self.item_embedding(extended) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        candidate_hidden = x[torch.arange(batch, device=x.device), candidate_pos]
        return self.score_head(candidate_hidden).squeeze(-1)

    def score_candidates(self, input_seq: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """input_seq: (batch, max_seq_len), lengths: (batch,), candidates: (batch, num_candidates)
        -> scores (batch, num_candidates), scoring every candidate against its session's context."""
        batch, num_candidates = candidates.shape
        expanded_input = input_seq.unsqueeze(1).expand(batch, num_candidates, -1).reshape(batch * num_candidates, -1)
        expanded_lengths = lengths.unsqueeze(1).expand(batch, num_candidates).reshape(-1)
        flat_candidates = candidates.reshape(-1)
        scores = self.forward(expanded_input, expanded_lengths, flat_candidates)
        return scores.view(batch, num_candidates)

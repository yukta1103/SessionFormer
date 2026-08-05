import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position i can only attend to
    positions <= i. Right-padded sequences don't need an extra padding mask
    here: since padding always comes after real content, a real query
    position can never attend to a padded key position."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (batch, num_heads, seq_len, head_dim)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class SASRecBlock(nn.Module):
    """Causal self-attention + position-wise FFN, each with a residual
    connection and post-sublayer LayerNorm."""

    def __init__(self, embedding_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = PositionwiseFeedForward(embedding_dim, ff_hidden_dim, dropout)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x + self.dropout(self.attn(x)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x

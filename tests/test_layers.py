import torch

from sessionformer.models.layers import CausalSelfAttention, PositionwiseFeedForward, SASRecBlock


def test_causal_attention_output_shape():
    attn = CausalSelfAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(4, 6, 8)
    out = attn(x)
    assert out.shape == (4, 6, 8)


def test_causal_attention_does_not_leak_future_positions():
    torch.manual_seed(0)
    attn = CausalSelfAttention(embedding_dim=8, num_heads=2)
    attn.eval()

    x = torch.randn(1, 5, 8)
    out_before = attn(x)

    x_changed = x.clone()
    x_changed[0, -1] = torch.randn(8)  # perturb only the last position
    out_after = attn(x_changed)

    # every position except the last must be unaffected by a change to the future
    assert torch.allclose(out_before[0, :-1], out_after[0, :-1], atol=1e-6)
    assert not torch.allclose(out_before[0, -1], out_after[0, -1])


def test_feedforward_output_shape():
    ffn = PositionwiseFeedForward(embedding_dim=8, hidden_dim=16)
    x = torch.randn(4, 6, 8)
    assert ffn(x).shape == (4, 6, 8)


def test_sasrec_block_output_shape():
    block = SASRecBlock(embedding_dim=8, num_heads=2, ff_hidden_dim=16)
    x = torch.randn(4, 6, 8)
    assert block(x).shape == (4, 6, 8)


def test_causal_attention_rejects_bad_head_count():
    try:
        CausalSelfAttention(embedding_dim=8, num_heads=3)
        assert False, "expected ValueError"
    except ValueError:
        pass

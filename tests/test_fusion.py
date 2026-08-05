import torch

from sessionformer.content.fusion import cosine_scores, fuse_scores, session_content_vector


def test_session_content_vector_averages_non_pad_items():
    item_embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])  # PAD, item1, item2, item3
    input_seq = torch.tensor([[1, 2, 0]])  # item1, item2, PAD
    vec = session_content_vector(item_embeddings, input_seq)
    assert torch.allclose(vec, torch.tensor([[0.5, 0.5]]))


def test_session_content_vector_all_padding_does_not_divide_by_zero():
    item_embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    input_seq = torch.tensor([[0, 0]])
    vec = session_content_vector(item_embeddings, input_seq)
    assert torch.allclose(vec, torch.tensor([[0.0, 0.0]]))


def test_cosine_scores_shape_and_range():
    query = torch.randn(3, 8)
    item_embeddings = torch.randn(10, 8)
    scores = cosine_scores(query, item_embeddings)
    assert scores.shape == (3, 10)
    assert scores.max() <= 1.0001 and scores.min() >= -1.0001


def test_fuse_scores_leaves_warm_items_as_normalized_collab_score():
    collab = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    content = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    is_cold = torch.tensor([False, False, True, True])
    fused = fuse_scores(collab, content, is_cold, alpha=0.5)

    collab_norm = (collab - collab.min()) / (collab.max() - collab.min())
    assert torch.allclose(fused[0, :2], collab_norm[0, :2])


def test_fuse_scores_blends_cold_items_toward_content():
    collab = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    content = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # item index 2 has max content score
    is_cold = torch.tensor([False, False, True, True])

    fused_low_alpha = fuse_scores(collab, content, is_cold, alpha=0.0)
    fused_high_alpha = fuse_scores(collab, content, is_cold, alpha=1.0)

    # alpha=0 on cold items should match pure normalized collaborative score
    collab_norm = (collab - collab.min()) / (collab.max() - collab.min())
    assert torch.allclose(fused_low_alpha[0, 2:], collab_norm[0, 2:])
    # alpha=1 on cold items should match pure normalized content score
    content_norm = (content - content.min()) / (content.max() - content.min())
    assert torch.allclose(fused_high_alpha[0, 2:], content_norm[0, 2:])

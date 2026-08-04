import torch

from sessionformer.negatives.sampler import UniformNegativeSampler


def test_sample_shape():
    sampler = UniformNegativeSampler(vocab_size=50, num_negatives=8)
    positive = torch.randint(2, 50, (4, 6))
    negatives = sampler.sample(positive)
    assert negatives.shape == (4, 6, 8)


def test_negatives_never_collide_with_positive():
    sampler = UniformNegativeSampler(vocab_size=10, num_negatives=5)
    positive = torch.full((100, 1), 3, dtype=torch.long)
    negatives = sampler.sample(positive)
    assert not (negatives == positive.unsqueeze(-1)).any()


def test_negatives_exclude_pad_and_unk():
    sampler = UniformNegativeSampler(vocab_size=20, num_negatives=50)
    positive = torch.zeros((10, 1), dtype=torch.long)  # PAD position
    negatives = sampler.sample(positive)
    assert (negatives >= 2).all()

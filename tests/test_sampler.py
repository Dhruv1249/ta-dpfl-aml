import numpy as np

from privacy.dp_sgd import poisson_sample_indices


def test_nonsecure_poisson_sampler_empirical_rate():
    indices = np.arange(20000)
    rng = np.random.default_rng(7)
    sampled = poisson_sample_indices(indices, 0.1, secure=False, rng=rng)
    assert abs(len(sampled) / len(indices) - 0.1) < 0.015
    assert len(np.unique(sampled)) == len(sampled)


def test_secure_sampler_has_valid_subset():
    indices = np.arange(500)
    sampled = poisson_sample_indices(indices, 0.2, secure=True)
    assert set(sampled).issubset(set(indices))
    assert len(np.unique(sampled)) == len(sampled)

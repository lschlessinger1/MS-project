import pytest
import numpy as np

from src.autoks.distance.sampling import sample
from src.autoks.distance.sampling.halton import HaltonSampler, GeneralizedHaltonSampler
from src.autoks.distance.sampling.sample import _create_sampler
from src.autoks.distance.sampling.sobol import SobolSampler


def test_sample_sobol():
    n, d = 3, 1
    samples = sample(sampler_type='sobol', n_samples=n, n_dims=d)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape == (n, d)
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_sample_halton():
    n, d = 3, 1
    samples = sample(sampler_type='halton', n_samples=n, n_dims=d)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape == (n, d)
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_sample_generalized_halton():
    n, d = 3, 1
    samples = sample(sampler_type='generalized_halton', n_samples=n, n_dims=d)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape == (n, d)
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_create_sampler_sobol():
    sampler = _create_sampler(sampler_type='sobol')
    assert isinstance(sampler, SobolSampler)


def test_create_sampler_halton():
    sampler = _create_sampler(sampler_type='halton')
    assert isinstance(sampler, HaltonSampler)


def test_create_sampler_generalized_halton():
    sampler = _create_sampler(sampler_type='generalized_halton')
    assert isinstance(sampler, GeneralizedHaltonSampler)


def test_create_sampler_unknown():
    with pytest.raises(ValueError):
        _create_sampler(sampler_type='foo')

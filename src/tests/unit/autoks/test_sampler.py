import pytest

from src.autoks.distance.sampling.sampler import Sampler


def test_create_empty_sampler():
    sampler = Sampler()
    assert isinstance(sampler, Sampler)


def test_create_sampler_with_seed():
    sampler = Sampler(seed=42)
    assert isinstance(sampler, Sampler)
    assert sampler.seed == 42


def test_sample():
    sampler = Sampler()
    with pytest.raises(NotImplementedError):
        sampler.sample(n_points=1, n_dims=1)

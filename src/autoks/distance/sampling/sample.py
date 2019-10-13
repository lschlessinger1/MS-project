import numpy as np

from src.autoks.distance.sampling.halton import HaltonSampler, GeneralizedHaltonSampler
from src.autoks.distance.sampling.sampler import Sampler
from src.autoks.distance.sampling.sobol import SobolSampler


def sample(sampler_type: str,
           n_samples: int,
           n_dims: int,
           *args,
           **kwargs) -> np.ndarray:
    """Sample from a variety of samplers."""
    sampler = _create_sampler(sampler_type, *args, **kwargs)
    return sampler.sample(n_points=n_samples, n_dims=n_dims)


def _create_sampler(sampler_type: str, *args, **kwargs) -> Sampler:
    """Create sampler from string."""
    if sampler_type == 'halton':
        return HaltonSampler(*args, **kwargs)
    elif sampler_type == 'generalized_halton':
        return GeneralizedHaltonSampler(*args, **kwargs)
    elif sampler_type == 'sobol':
        return SobolSampler(*args, **kwargs)
    else:
        raise ValueError(f'Sampler type {sampler_type} not found.')

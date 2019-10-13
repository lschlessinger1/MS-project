import ghalton
import numpy as np

from src.autoks.distance.sampling.sampler import Sampler
from src.autoks.distance.sampling.scramble import scramble_array


def generate_halton(n: int, d: int):
    sequencer = ghalton.Halton(d)
    return sequencer.get(n)


def generate_generalized_halton(n: int, d: int):
    sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:d])
    return sequencer.get(n)


def halton_sample(n_samples: int,
                  n_dims: int,
                  leap: int = 100,
                  scramble: bool = True):
    # Generate samples
    n_gen = n_samples * leap if leap > 0 else n_samples

    samples = generate_halton(n_gen, n_dims)

    # Apply leap
    if leap > 0:
        samples = samples[::leap]

    # Scramble
    if scramble:
        scramble_array(samples)

    return samples


def generalized_halton_sample(n_samples: int,
                              n_dims: int):
    max_dims = 100

    if n_dims > max_dims:
        raise ValueError(
            f'{ghalton.GeneralizedHalton.__class__.__name__} supports up to {max_dims} spatial dimensions.')

    return generate_generalized_halton(n_samples, n_dims)


class HaltonSampler(Sampler):
    """Halton sequence sampler."""

    def sample(self, n_points: int, n_dims: int) -> np.ndarray:
        return np.asarray(halton_sample(n_points, n_dims))


class GeneralizedHaltonSampler(Sampler):
    """Generalized Halton sequence sampler"""

    def sample(self, n_points: int, n_dims: int) -> np.ndarray:
        return np.asarray(generalized_halton_sample(n_points, n_dims))

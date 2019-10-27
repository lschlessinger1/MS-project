import numpy as np
from sobol_seq import i4_sobol_generate

from src.autoks.distance.sampling.sampler import Sampler
from src.autoks.distance.sampling.scramble import scramble_array

# FIXME: This should actually accept up to 1111 dimensions
MAX_DIMS = 40


def gen_sobol(n: int,
              d: int,
              skip: int = 0):
    """Wrapper for i4_sobol_generate

    :param n:
    :param d:
    :param skip:
    :return:
    """
    # Slight bug:
    # Generated samples do not include 0 as first in sequence.
    if skip == 0:
        sample_0 = np.zeros(d)
        samples = np.vstack((sample_0, i4_sobol_generate(d, n - 1, skip=skip)))
    else:
        samples = i4_sobol_generate(d, n, skip=skip)
    return samples


def sobol_sample(n_samples: int,
                 n_dims: int,
                 skip: int = 1000,
                 scramble: bool = True) -> np.ndarray:
    """Get samples from a Sobol sequence.

    n_samples: The number of points in the sequence.
    n_dims: Number of dimensions in the set.
    skip: Number of initial points to omit.
    """
    if n_dims > MAX_DIMS:
        raise ValueError(f'sobol_seq supports up to {MAX_DIMS} spatial dimensions.')

    # Generate samples
    samples = gen_sobol(n_samples, n_dims, skip=skip)

    # Scramble
    if scramble:
        scramble_array(samples)

    return samples


class SobolSampler(Sampler):
    def sample(self, n_points: int, n_dims: int) -> np.ndarray:
        return sobol_sample(n_points, n_dims)

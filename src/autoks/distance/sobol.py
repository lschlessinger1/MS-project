import numpy as np

from sobol_seq import i4_sobol_generate

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
                 leap: int = 100,
                 scramble: bool = True) -> np.ndarray:
    """Get samples from a Sobol sequence.

    n_samples: The number of points in the sequence.
    n_dims: Number of dimensions in the set.
    skip: Number of initial points to omit.
    leap: Number of points to miss out between returned points.
    """
    if n_dims > MAX_DIMS:
        raise ValueError(f'sobol_seq supports up to {MAX_DIMS} spatial dimensions.')

    # Generate samples
    n_gen = n_samples * leap if leap > 0 else n_samples

    samples = gen_sobol(n_gen, n_dims, skip=skip)

    # Apply leap
    if leap > 0:
        samples = samples[::leap]

    # Scramble
    if scramble:
        scramble_array(samples)

    return samples


def scramble_array(a: np.ndarray) -> None:
    """In-place scrambling of an array."""
    # FIXME: This should actually use the Matousek-Affine-Owen scrambling algorithm
    # Matousek, J. “On the L2-Discrepancy for Anchored Boxes.” Journal of Complexity.
    # Vol. 14, Number 4, 1998, pp. 527–556.
    # for now, use a Naive scrambling method
    np.random.shuffle(a)

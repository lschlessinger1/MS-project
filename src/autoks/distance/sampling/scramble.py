import numpy as np


def scramble_array(a: np.ndarray) -> None:
    """In-place scrambling of an array."""
    # FIXME: This should actually use the Matousek-Affine-Owen scrambling algorithm
    # Matousek, J. “On the L2-Discrepancy for Anchored Boxes.” Journal of Complexity.
    # Vol. 14, Number 4, 1998, pp. 527–556.
    # for now, use a Naive scrambling method
    np.random.shuffle(a)

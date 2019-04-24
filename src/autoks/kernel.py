from typing import List, Union

import numpy as np
from scipy.spatial.distance import cdist, pdist

from src.autoks.backend.kernel import RawKernelType, subkernel_expression, kernel_l2_dist, kernel_to_infix, \
    kernels_to_kernel_vecs
from src.autoks.symbolic.kernel_symbol import KernelSymbol
from src.autoks.util import remove_duplicates


# Symbolic interface
def tokens_to_kernel_symbols(tokens: List[Union[str, RawKernelType]]) -> List[Union[str, KernelSymbol]]:
    symbols = []
    for token in tokens:
        if isinstance(token, str):
            symbols.append(token)
        elif isinstance(token, RawKernelType):
            name = subkernel_expression(token)
            symbols.append(KernelSymbol(name, token))
    return symbols


def covariance_distance(kernels: List[RawKernelType],
                        x: np.ndarray) -> np.ndarray:
    """Euclidean distance of all pairs kernels.

    :param kernels:
    :param x:
    :return:
    """
    # For each pair of kernel matrices, compute Euclidean distance
    n_kernels = len(kernels)
    dists = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(i + 1, n_kernels):
            dists[i, j] = kernel_l2_dist(kernels[i], kernels[j], x)
    # Make symmetric
    dists = (dists + dists.T) / 2.
    return dists


def remove_duplicate_kernels(kernels: List[RawKernelType]) -> List[RawKernelType]:
    """Remove duplicate kernels.

    :param kernels:
    :return:
    """
    return remove_duplicates([kernel_to_infix(k) for k in kernels], kernels)


def kernel_vec_avg_dist(kvecs1: np.ndarray,
                        kvecs2: np.ndarray) -> float:
    """Average Euclidean distance between two lists of vectors.

    :param kvecs1: n_1 x d array encoding of an additive kernel part
    :param kvecs2: n_2 x d array encoding of an additive kernel part
    :return:
    """
    dists = cdist(kvecs1, kvecs2, metric='euclidean')
    return float(np.mean(dists))


def all_pairs_avg_dist(kernels: List[RawKernelType],
                       base_kernels: List[str],
                       n_dims: int) -> float:
    """Mean distance between all pairs of kernels.

    Can be thought of as a diversity score of a population of kernels
    :param kernels:
    :param base_kernels:
    :param n_dims:
    :return:
    """
    kernel_vecs = kernels_to_kernel_vecs(kernels, base_kernels, n_dims)

    # compute average Euclidean distance for all pairs of kernels
    data = np.empty((len(kernel_vecs), 1), dtype=np.object)
    for i, kvec in enumerate(kernel_vecs):
        data[i, 0] = kvec
    pairwise_dist = pdist(data, metric=lambda u, v: kernel_vec_avg_dist(u[0], v[0]))
    return float(np.mean(pairwise_dist))

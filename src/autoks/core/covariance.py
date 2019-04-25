from typing import Optional, List, Union

import numpy as np
from graphviz import Source
from scipy.spatial.distance import cdist, pdist
from sympy import pprint, latex, mathml, dotprint

from src.autoks.backend.kernel import RawKernelType, kernel_to_infix_tokens, tokens_to_str, sort_kernel, additive_form, \
    is_base_kernel, subkernel_expression, kernel_l2_dist, kernels_to_kernel_vecs, is_prod_kernel, is_sum_kernel
from src.autoks.core.kernel_encoding import kernel_to_tree
from src.autoks.symbolic.kernel_symbol import KernelSymbol
from src.autoks.symbolic.util import postfix_tokens_to_symbol
from src.autoks.util import remove_duplicates
from src.evalg.encoding import infix_tokens_to_postfix_tokens
from src.test.autoks.test_kernel_encoding import KernelTree


class Covariance:
    """A wrapper for a GPy Kern"""

    def __init__(self, kernel: RawKernelType):
        if not isinstance(kernel, RawKernelType):
            raise TypeError(f'kernel must be {RawKernelType.__name__}. Found type {kernel.__class__.__name__}.')
        self.raw_kernel = kernel

    @property
    def raw_kernel(self) -> RawKernelType:
        return self._raw_kernel

    @raw_kernel.setter
    def raw_kernel(self, new_kernel: RawKernelType) -> None:
        self._raw_kernel = new_kernel
        # Set other raw_kernel parameters
        self.infix_tokens = kernel_to_infix_tokens(self.raw_kernel)
        self.postfix_tokens = infix_tokens_to_postfix_tokens(self.infix_tokens)
        self.infix = tokens_to_str(self.infix_tokens, show_params=False)
        self.infix_full = tokens_to_str(self.infix_tokens, show_params=True)
        self.postfix = tokens_to_str(self.postfix_tokens, show_params=False)
        postfix_token_symbols = tokens_to_kernel_symbols(self.postfix_tokens)
        self.symbolic_expr = postfix_tokens_to_symbol(postfix_token_symbols)
        self.symbolic_expr_expanded = self.symbolic_expr.expand()

    def to_binary_tree(self) -> KernelTree:
        """Get the binary tree representation of the kernel

        :return:
        """
        return kernel_to_tree(self.raw_kernel)

    def canonical(self) -> RawKernelType:
        return sort_kernel(self.raw_kernel)

    def to_additive_form(self) -> RawKernelType:
        """Convert the kernel to additive form.

        :return:
        """
        return additive_form(self.raw_kernel)

    def pretty_print(self) -> None:
        """Pretty print the kernel.

        :return:
        """
        pprint(self.symbolic_expr)

    def print_full(self) -> None:
        """Print the verbose version of the kernel.

        :return:
        """
        print(self.infix_full)

    def is_base(self) -> bool:
        return is_base_kernel(self.raw_kernel)

    def is_sum(self):
        return is_sum_kernel(self.raw_kernel)

    def is_prod(self):
        return is_prod_kernel(self.raw_kernel)

    def priors(self) -> Optional:
        raise NotImplementedError('This will be implemented soon')

    def symbolically_equals(self, other):
        return self.symbolic_expr == other.symbolic_expr

    def symbolic_expanded_equals(self, other):
        return self.symbolic_expr_expanded == other.symbolic_expr_expanded

    def infix_equals(self, other):
        # naively compare based on infix
        return isinstance(other, Covariance) and other.infix == self.infix

    def as_latex(self) -> str:
        return latex(self.symbolic_expr)

    def as_mathml(self) -> str:
        return mathml(self.symbolic_expr)

    def as_dot(self) -> str:
        return dotprint(self.symbolic_expr)

    def as_graph(self) -> Source:
        return Source(self.as_dot())

    def __add__(self, other):
        return Covariance(self.raw_kernel + other.raw_kernel)

    def __mul__(self, other):
        return Covariance(self.raw_kernel * other.raw_kernel)

    def __str__(self):
        return str(self.symbolic_expr)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={self.infix_full !r})'


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


def covariance_distance(covariances: List[Covariance],
                        x: np.ndarray) -> np.ndarray:
    """Euclidean distance of all pairs gp_models.

    :param covariances:
    :param x:
    :return:
    """
    # For each pair of kernel matrices, compute Euclidean distance
    n_kernels = len(covariances)
    dists = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(i + 1, n_kernels):
            dists[i, j] = kernel_l2_dist(covariances[i].raw_kernel, covariances[j].raw_kernel, x)
    # Make symmetric
    dists = (dists + dists.T) / 2.
    return dists


def remove_duplicate_kernels(covariances: List[Covariance]) -> List[Covariance]:
    """Remove duplicate gp_models.

    :param covariances:
    :return:
    """
    return remove_duplicates([cov.symbolic_expr for cov in covariances], covariances)


def kernel_vec_avg_dist(kvecs1: np.ndarray,
                        kvecs2: np.ndarray) -> float:
    """Average Euclidean distance between two lists of vectors.

    :param kvecs1: n_1 x d array encoding of an additive kernel part
    :param kvecs2: n_2 x d array encoding of an additive kernel part
    :return:
    """
    dists = cdist(kvecs1, kvecs2, metric='euclidean')
    return float(np.mean(dists))


def all_pairs_avg_dist(covariances: List[Covariance],
                       base_kernels: List[str],
                       n_dims: int) -> float:
    """Mean distance between all pairs of gp_models.

    Can be thought of as a diversity score of a population of gp_models
    :param covariances:
    :param base_kernels:
    :param n_dims:
    :return:
    """
    raw_kernels = [cov.raw_kernel for cov in covariances]
    kernel_vecs = kernels_to_kernel_vecs(raw_kernels, base_kernels, n_dims)

    # compute average Euclidean distance for all pairs of gp_models
    data = np.empty((len(kernel_vecs), 1), dtype=np.object)
    for i, kvec in enumerate(kernel_vecs):
        data[i, 0] = kvec
    pairwise_dist = pdist(data, metric=lambda u, v: kernel_vec_avg_dist(u[0], v[0]))
    return float(np.mean(pairwise_dist))

from typing import Optional, List, Union

import numpy as np
from GPy.kern import Kern
from GPy.kern.src.kern import CombinationKernel
from graphviz import Source
from scipy.spatial.distance import cdist, pdist
from sympy import pprint, latex, mathml, dotprint

from src.autoks.backend.kernel import RawKernelType, kernel_to_infix_tokens, tokens_to_str, sort_kernel, additive_form, \
    is_base_kernel, subkernel_expression, kernels_to_kernel_vecs, is_prod_kernel, is_sum_kernel, compute_kernel, \
    KERNEL_DICT, set_priors
from src.autoks.core.hyperprior import HyperpriorMap
from src.autoks.core.kernel_encoding import kernel_to_tree, KernelTree
from src.autoks.symbolic.kernel_symbol import KernelSymbol
from src.autoks.symbolic.util import postfix_tokens_to_symbol
from src.autoks.util import remove_duplicates
from src.evalg.encoding import infix_tokens_to_postfix_tokens
from src.evalg.serialization import Serializable


class Covariance(Serializable):
    """A wrapper for a GPy Kern"""

    def __init__(self, kernel: RawKernelType):
        self.raw_kernel = kernel

    @property
    def raw_kernel(self) -> RawKernelType:
        return self._raw_kernel

    @raw_kernel.setter
    def raw_kernel(self, new_kernel: RawKernelType) -> None:
        if not isinstance(new_kernel, RawKernelType):
            raise TypeError(f'kernel must be {RawKernelType.__name__}. Found type {new_kernel.__class__.__name__}.')
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

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["kernel"] = self.raw_kernel.to_dict()
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        input_dict["kernel"] = Kern.from_dict(input_dict["kernel"])
        return input_dict

    def to_binary_tree(self) -> KernelTree:
        """Get the binary tree representation of the kernel

        :return:
        """
        return kernel_to_tree(self.raw_kernel)

    def canonical(self) -> RawKernelType:
        """Get canonical form of backend kernel.

        :return:
        """
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
        """Determine whether backend kernel is a 1-d base kernel."""
        return is_base_kernel(self.raw_kernel)

    def is_sum(self) -> bool:
        """Determine whether backend kernel is a sum kernel."""
        return is_sum_kernel(self.raw_kernel)

    def is_prod(self) -> bool:
        """Determine whether backend kernel is a product kernel."""
        return is_prod_kernel(self.raw_kernel)

    def priors(self) -> Optional:
        """Get the priors of the kernel."""
        raise NotImplementedError('This will be implemented soon')

    def set_hyperpriors(self, hyperpriors: HyperpriorMap) -> None:
        inv_KERNEL_DICT = {v: k for k, v in KERNEL_DICT.items()}

        def set_kern_prior(x):
            if not isinstance(x, CombinationKernel) and isinstance(x, Kern):
                cls_name = inv_KERNEL_DICT[x.__class__]
                set_priors(x, hyperpriors[cls_name], in_place=True)

        for part in self.infix_tokens:
            set_kern_prior(part)

    def symbolically_equals(self, other) -> bool:
        """Determine whether this covariance's kernel expression is the same as another's kernel expression."""
        return self.symbolic_expr == other.symbolic_expr

    def symbolic_expanded_equals(self, other) -> bool:
        """Determine whether this covariance's expanded kernel expression is the same as another's expanded kernel
        expression."""
        return self.symbolic_expr_expanded == other.symbolic_expr_expanded

    def infix_equals(self, other) -> bool:
        """Determine whether this covariance's kernel infix expression is the same as another's infix kernel
        expression."""
        # naively compare based on infix
        return isinstance(other, Covariance) and other.infix == self.infix

    def as_latex(self) -> str:
        """Get a LaTeX representation of this covariance."""
        return latex(self.symbolic_expr)

    def as_mathml(self) -> str:
        """Get a MathML representation of this covariance."""
        return mathml(self.symbolic_expr)

    def as_dot(self) -> str:
        """Get a DOT representation of this covariance."""
        return dotprint(self.symbolic_expr)

    def as_graph(self) -> Source:
        """Get a GraphViz Source representation of this covariance."""
        return Source(self.as_dot())

    def __add__(self, other):
        return Covariance(self.raw_kernel + other.raw_kernel)

    def __mul__(self, other):
        return Covariance(self.raw_kernel * other.raw_kernel)

    def __str__(self):
        return str(self.symbolic_expr)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={self.infix_full !r})'


def pretty_print_covariances(covariances: List[Covariance],
                             kernel_type_label: Optional[str] = None):
    """Pretty print a list of covariances."""
    n_kernels = len(covariances)

    plural_suffix = 's' if n_kernels > 1 else ''
    ending = f'kernel{plural_suffix}:'
    if kernel_type_label is not None:
        message = f'{n_kernels} {kernel_type_label} {ending}'
    else:
        message = f'{n_kernels} {ending}'
    message = message.capitalize()
    print(message)
    for cov in covariances:
        cov.pretty_print()
    print('')


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


def euclidean_distance(x: np.ndarray,
                       y: np.ndarray) -> float:
    return np.linalg.norm(x - y)


def kernel_l2_dist(kernel_1: RawKernelType,
                   kernel_2: RawKernelType,
                   x: np.ndarray) -> float:
    """Euclidean distance between two kernel matrices.

    :param kernel_1:
    :param kernel_2:
    :param x:
    :return:
    """

    return euclidean_distance(compute_kernel(kernel_1, x), compute_kernel(kernel_2, x))


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
    if len(covariances) < 2:
        return 0.

    raw_kernels = [cov.raw_kernel for cov in covariances]
    kernel_vecs = kernels_to_kernel_vecs(raw_kernels, base_kernels, n_dims)

    # compute average Euclidean distance for all pairs of gp_models
    data = np.empty((len(kernel_vecs), 1), dtype=np.object)
    for i, kvec in enumerate(kernel_vecs):
        data[i, 0] = kvec
    pairwise_dist = pdist(data, metric=lambda u, v: kernel_vec_avg_dist(u[0], v[0]))
    return float(np.mean(pairwise_dist))


def inner_frob(m, n):
    """Frobenius inner product"""
    return np.trace(m.T.conjugate() @ n)


def alignment(k1: np.ndarray, k2: np.ndarray) -> float:
    """Alignment A(k1, k2) between two kernel matrices

    It can be viewed as the cosine of the angle between the matrices viewed as 2-d vectors

    0 <= A(k1, k2) <= 1

        Alignment $A$ between two kernel matrices $K_1$ and $K_2$:

    $$A(K_1, K_2) = \frac{\langle K_1, K_2 \rangle_F}{\sqrt{\langle K_1, K_1 \rangle_F \langle K_2, K_2 \rangle_F}}$$
    """
    k1_dot_k2 = inner_frob(k1, k2)
    k1_dot_k1 = inner_frob(k1, k1)
    k2_dot_k2 = inner_frob(k2, k2)

    return k1_dot_k2 / np.sqrt(k1_dot_k1 * k2_dot_k2)


def centered_alignment(k1: np.ndarray, k2: np.ndarray) -> float:
    """Centered kernel alignment

    Cortes et al. (2012)
    """
    k1_centered = center_kernel(k1)
    k2_centered = center_kernel(k2)
    return alignment(k1_centered, k2_centered)


def center_kernel(k: np.ndarray) -> np.ndarray:
    """Center a kernel matrix"""
    m = k.shape[0]
    identity = np.eye(m)
    ones = np.ones((m, 1))
    centering = (identity - (ones @ ones.T) / m)
    return centering @ k @ centering


def pairwise_centered_alignments(covariances: List[Covariance],
                                 x: np.ndarray) -> np.ndarray:
    """Alignment of all pairs of covariances.

    :param covariances:
    :param x:
    :return:
    """
    # For each pair of kernel matrices, compute alignment
    n_kernels = len(covariances)
    dists = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(i + 1, n_kernels):
            k1 = compute_kernel(covariances[i].raw_kernel, x)
            k2 = compute_kernel(covariances[j].raw_kernel, x)
            dists[i, j] = centered_alignment(k1, k2)
    # Make symmetric
    dists = (dists + dists.T) / 2.
    return dists

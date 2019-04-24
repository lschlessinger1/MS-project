from typing import Optional, List

import numpy as np
from GPy.kern import Kern
from sympy import pprint

from src.autoks.kernel import kernel_to_infix_tokens, tokens_to_str, tokens_to_kernel_symbols, KernelTree, \
    kernel_to_tree, additive_form, kernel_to_infix, encode_kernel, get_priors, encode_prior
from src.autoks.symbolic.util import postfix_tokens_to_symbol
from src.autoks.util import remove_duplicates
from src.evalg.encoding import infix_tokens_to_postfix_tokens


class GPModel:
    """GPy kernel wrapper
    """
    _kernel: Kern
    lik_params: Optional[np.ndarray]
    evaluated: bool
    nan_scored: bool
    expanded: bool

    def __init__(self, kernel, lik_params=None, evaluated=False, nan_scored=False, expanded=False):
        self.kernel = kernel
        self.lik_params = lik_params
        self.evaluated = evaluated
        self.nan_scored = nan_scored
        self.expanded = expanded
        self._score = None

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score
        # Update evaluated as well
        self.evaluated = True

    @property
    def kernel(self) -> Kern:
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: Kern) -> None:
        self._kernel = new_kernel
        # Set other kernel parameters
        self.infix_tokens = kernel_to_infix_tokens(self.kernel)
        self.postfix_tokens = infix_tokens_to_postfix_tokens(self.infix_tokens)
        self.infix = tokens_to_str(self.infix_tokens, show_params=False)
        self.postfix = tokens_to_str(self.postfix_tokens, show_params=False)
        postfix_token_symbols = tokens_to_kernel_symbols(self.postfix_tokens)
        self.symbolic_expr = postfix_tokens_to_symbol(postfix_token_symbols)
        self.symbolic_expr_expanded = self.symbolic_expr.expand()

    def to_binary_tree(self) -> KernelTree:
        """Get the binary tree representation of the kernel

        :return:
        """
        return kernel_to_tree(self.kernel)

    def to_additive_form(self) -> None:
        """Convert the kernel to additive form.

        :return:
        """
        self.kernel = additive_form(self.kernel)

    def pprint_expr(self) -> None:
        pprint(self.symbolic_expr)

    def pretty_print(self) -> None:
        """Pretty print the kernel.

        :return:
        """
        # print(str(self))
        self.pprint_expr()

    def print_full(self) -> None:
        """Print the verbose version of the kernel.

        :return:
        """
        infix_full = tokens_to_str(self.infix_tokens, show_params=True)
        print(infix_full)

    def __str__(self):
        return str(self.symbolic_expr)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={kernel_to_infix(self.kernel, show_params=True)!r}, ' \
            f'lik_params={self.lik_params!r}, evaluated={self.evaluated!r}, nan_scored={self.nan_scored!r}, ' \
            f'expanded={self.expanded!r}, score={self.score!r}) '


def pretty_print_gp_models(gp_models: List[GPModel],
                           kernel_type_label: Optional[str] = None):
    n_kernels = len(gp_models)

    plural_suffix = 's' if n_kernels > 1 else ''
    ending = f'kernel{plural_suffix}:'
    if kernel_type_label is not None:
        message = f'{n_kernels} {kernel_type_label} {ending}'
    else:
        message = f'{n_kernels} {ending}'
    message = message.capitalize()
    print(message)
    for k in gp_models:
        k.pretty_print()
    print('')


def remove_duplicate_gp_models(kernels: List[GPModel]) -> List[GPModel]:
    """Remove duplicate GPModel's.

    prioritizing when removing duplicates
        1. highest score
        2. not nan scored
        3. evaluated

    :param kernels:
    :return:
    """

    unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated and not kernel.nan_scored]
    evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
    nan_scored_kernels = [kernel for kernel in kernels if kernel.nan_scored]
    # Prioritize highest scoring kernel for duplicates
    sorted_evaluated_kernels = sorted(evaluated_kernels, key=lambda k: k.score, reverse=True)

    # Assume precedence by order.
    gp_models = sorted_evaluated_kernels + nan_scored_kernels + unevaluated_kernels
    return remove_duplicates([kernel_to_infix(gp_model.kernel) for gp_model in gp_models], gp_models)


def encode_gp_model(gp_model: GPModel) -> List[str]:
    """Encode GPModel

    :param gp_model:
    :return:
    """
    try:
        prior_enc = [encode_prior(prior) for prior in get_priors(gp_model.kernel)]
    except ValueError:
        prior_enc = None
    return [encode_kernel(gp_model.kernel), [prior_enc]]


def encode_gp_models(gp_models: List[GPModel]) -> np.ndarray:
    """Encode a list of models.

    :param gp_models: the models to encode.
    :return: An array containing encodings of the kernels.
    """
    enc = np.empty((len(gp_models), 1), dtype=np.object)
    for i, gp_model in enumerate(gp_models):
        enc[i, 0] = encode_gp_model(gp_model)
    return enc

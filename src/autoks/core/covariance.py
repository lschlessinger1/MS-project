from typing import Optional

from graphviz import Source
from sympy import pprint, latex, mathml, dotprint

from src.autoks.backend.kernel import RawKernelType, kernel_to_infix_tokens, tokens_to_str, sort_kernel, additive_form, \
    is_base_kernel
from src.autoks.core.kernel import tokens_to_kernel_symbols
from src.autoks.core.kernel_encoding import KernelTree, kernel_to_tree
from src.autoks.symbolic.util import postfix_tokens_to_symbol
from src.evalg.encoding import infix_tokens_to_postfix_tokens


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
        return f'{self.__class__.__name__}('f'kernel={self.infix_full !r}) '

    # TODO: test and add latex, mathml, dotprint (Source, etc.)

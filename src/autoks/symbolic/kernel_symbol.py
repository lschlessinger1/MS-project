from GPy.kern import Kern
from sympy import Symbol


class KernelSymbol(Symbol):
    """Simple wrapper for Symbol that stores a 1-D kernel."""

    def __new__(cls, name: str, kernel_one_d: Kern):
        obj = Symbol.__new__(cls, name)
        obj.kernel_one_d = kernel_one_d
        return obj

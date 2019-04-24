from collections import Counter
from typing import Iterable

from GPy.kern import Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern


def counter_repr(x: Iterable) -> Counter:
    """Convert iterable to a Counter.

    :param x:
    :return:
    """
    return Counter(frozenset(Counter(item).items()) for item in x)


def lists_equal_without_order(a: list, b: list) -> bool:
    """Test if two lists are equal to each other without considering the order.

    :param a:
    :param b:
    :return:
    """
    return counter_repr(a) == counter_repr(b)


def same_combo_type(k1: Kern, k2: Kern) -> bool:
    """Test if two kernels are the same type of combination kernel.

    :param k1:
    :param k2:
    :return:
    """
    return isinstance(k1, Add) and isinstance(k2, Add) or isinstance(k1, Prod) and isinstance(k2, Prod)


def base_kernel_eq(kern_1: Kern, kern_2: Kern) -> bool:
    same_type = kern_1.name == kern_2.name
    same_dim = len(kern_1.active_dims) == 1 and kern_1.active_dims[0] == kern_2.active_dims[0]
    return same_type and same_dim


def has_combo_kernel_type(kernels: Iterable[Kern], kern: Kern) -> bool:
    """Test if an iterable of kernels are the same combination kernel type as another kernel.

    :param kernels:
    :param kern:
    :return:
    """
    is_combo_kernel = isinstance(kern, CombinationKernel)
    is_base_kernel = isinstance(kern, Kern) and not is_combo_kernel
    for kernel in kernels:
        if isinstance(kernel, CombinationKernel) and is_combo_kernel:
            k_parts = [(k.__class__, k.active_dims[0]) for k in kern.parts]
            part_list = [(part.__class__, part.active_dims[0]) for part in kernel.parts]
            same_combo = same_combo_type(kernel, kern)
            if lists_equal_without_order(k_parts, part_list) and same_combo:
                return True
        elif isinstance(kernel, Kern) and is_base_kernel:
            if base_kernel_eq(kernel, kern):
                return True
    return False

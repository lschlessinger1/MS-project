from collections import Counter

from GPy.kern import Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern


def counter_repr(x):
    return Counter(frozenset(Counter(item).items()) for item in x)


def lists_equal_without_order(a, b):
    return counter_repr(a) == counter_repr(b)


def same_combo_type(k1, k2):
    return isinstance(k1, Add) and isinstance(k2, Add) or isinstance(k1, Prod) and isinstance(k2, Prod)


def has_combo_kernel_type(kernels, kern):
    is_combo_kernel = isinstance(kern, CombinationKernel)
    is_base_kernel = isinstance(kern, Kern) and not is_combo_kernel
    for kernel in kernels:
        if isinstance(kernel, CombinationKernel) and is_combo_kernel:
            kparts = [(k.__class__, k.active_dims[0]) for k in kern.parts]
            part_list = [(part.__class__, part.active_dims[0]) for part in kernel.parts]
            same_combo = same_combo_type(kernel, kern)
            if lists_equal_without_order(kparts, part_list) and same_combo:
                return True
        elif isinstance(kernel, Kern) and is_base_kernel:
            same_type = kernel.name == kern.name
            same_dim = len(kernel.active_dims) == 1 and kernel.active_dims[0] == kern.active_dims[0]
            if same_type and same_dim:
                return True
    return False

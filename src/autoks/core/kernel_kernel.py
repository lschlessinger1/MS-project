from typing import List

from GPy.kern import RBFKernelKernel
from numpy import ndarray

from src.autoks.distance.metrics import k_vec_metric, shd_metric, euclidean_metric, hellinger_metric


def k_vec_kernel_kernel(base_kernels: List[str],
                        n_dims: int,
                        **kwargs) -> RBFKernelKernel:
    """Construct a kernel kernel using the kernel vector metric.

    :param n_dims:
    :param base_kernels:
    :param kwargs: KernelKernel keyword arguments.
    :return: The K_vec kernel kernel
    """
    input_dict = {
        'base_kernels': base_kernels,
        'n_dims': n_dims
    }
    return RBFKernelKernel(distance_metric=k_vec_metric, dm_kwargs_dict=input_dict, **kwargs)


def shd_kernel_kernel(**kwargs) -> RBFKernelKernel:
    """Construct a kernel kernel using the SHD metric.

    :param kwargs: KernelKernel keyword arguments.
    :return: The SHD kernel kernel
    """
    return RBFKernelKernel(distance_metric=shd_metric, **kwargs)


def euclidean_kernel_kernel(x_train: ndarray, **kwargs) -> RBFKernelKernel:
    """Construct a kernel kernel using the SHD metric.

    :param x_train: Training data
    :param kwargs: KernelKernel keyword arguments.
    :return: The SHD kernel kernel
    """
    input_dict = {
        'get_x_train': lambda: x_train
    }
    return RBFKernelKernel(distance_metric=euclidean_metric, dm_kwargs_dict=input_dict, **kwargs)


def hellinger_kernel_kernel(x_train: ndarray, **kwargs) -> RBFKernelKernel:
    """

    :param x_train:
    :param kwargs:
    :return:
    """
    input_dict = {
        'get_x_train': lambda: x_train
    }
    return RBFKernelKernel(distance_metric=hellinger_metric, dm_kwargs_dict=input_dict, **kwargs)

from typing import List, Callable

import numpy as np
from GPy.core.parameterization.priors import Gaussian

from src.autoks.distance.distance import HellingerDistanceBuilder
from src.autoks.kernel import decode_kernel, all_pairs_avg_dist, kernel_to_tree, hd_kern_nodes, kernel_l2_dist, \
    AKSKernel, set_priors, decode_prior
from src.evalg.fitness import structural_hamming_dist


def k_vec_metric(u: np.ndarray,
                 v: np.ndarray,
                 base_kernels: List[str],
                 n_dims: int) -> float:
    """Kernel vector encoding distance metric

    :param u:
    :param v:
    :param base_kernels:
    :param n_dims:
    :return:
    """
    aks_kernel_1_enc, aks_kernel_2_enc = u[0], v[0]
    k1, k2 = decode_kernel(aks_kernel_1_enc[0]), decode_kernel(aks_kernel_2_enc[0])
    return all_pairs_avg_dist([k1, k2], base_kernels, n_dims)


def shd_metric(u: np.ndarray,
               v: np.ndarray) -> float:
    """Structural hamming distance (SHD) metric

    :param u: An array containing the first list of an encoded kernel as its only element
    :param v: An array containing the second list of an encoded kernel as its only element
    :return: SHD between u and v
    """
    aks_kernel_1_enc, aks_kernel_2_enc = u[0], v[0]
    k1, k2 = decode_kernel(aks_kernel_1_enc[0]), decode_kernel(aks_kernel_2_enc[0])
    tree_1, tree_2 = kernel_to_tree(k1), kernel_to_tree(k2)
    return structural_hamming_dist(tree_1, tree_2, hd=hd_kern_nodes)


# Euclidean distance metric
def euclidean_metric(u: np.ndarray,
                     v: np.ndarray,
                     get_x_train: Callable[[], np.ndarray]) -> float:
    """Euclidean distance metric

    :param u: An array containing the first list of an encoded kernel as its only element
    :param v: An array containing the second list of an encoded kernel as its only element
    :param get_x_train:
    :return: Euclidean distance between u and v
    """
    aks_kernel_1_enc, aks_kernel_2_enc = u[0], v[0]
    k1, k2 = decode_kernel(aks_kernel_1_enc[0]), decode_kernel(aks_kernel_2_enc[0])
    x_train = get_x_train()
    return kernel_l2_dist(k1, k2, x_train)


# Hellinger distance metric
def hellinger_metric(u: np.ndarray,
                     v: np.ndarray,
                     get_x_train: Callable[[], np.ndarray]) -> float:
    aks_kernel_1_enc, aks_kernel_2_enc = u[0], v[0]
    kern_1, kern_2 = decode_kernel(aks_kernel_1_enc[0]), decode_kernel(aks_kernel_2_enc[0])

    has_priors = aks_kernel_1_enc[1] is not None and aks_kernel_2_enc[1] is not None
    if has_priors:
        priors_1 = [[decode_prior(enc) for enc in encs] for encs in aks_kernel_1_enc[1]]
        priors_2 = [[decode_prior(enc) for enc in encs] for encs in aks_kernel_2_enc[1]]
        prior_dict_1 = dict(zip(kern_1.parameter_names(), priors_1[0]))
        prior_dict_2 = dict(zip(kern_2.parameter_names(), priors_2[0]))
        kern_1 = set_priors(kern_1, prior_dict_1)
        kern_2 = set_priors(kern_2, prior_dict_2)

    x_train = get_x_train()

    noise_prior = Gaussian(np.log(0.01), 1)
    active_models = [AKSKernel(kern_1), AKSKernel(kern_2)]
    num_samples = 20
    max_num_hyperparameters = 40
    max_num_kernels = 1000
    initial_model_indices = [0, 1]
    builder = HellingerDistanceBuilder(noise_prior, num_samples, max_num_hyperparameters, max_num_kernels,
                                       active_models, initial_model_indices, data_X=x_train)
    builder.compute_distance(active_models, [0], [1])
    return builder._average_distance[0, 0]
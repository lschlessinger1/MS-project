from typing import List, Callable, Any, Union

import numpy as np

from src.autoks.backend.kernel import KERNEL_DICT, n_base_kernels, count_kernel_types
from src.autoks.core.covariance import all_pairs_avg_dist, pairwise_centered_alignments
from src.autoks.core.gp_model import GPModel
from src.autoks.statistics import StatBook


def update_stat_book(stat_book: StatBook,
                     gp_models: List[GPModel],
                     x_train,
                     base_kernel_names: List[str],
                     n_dims: int) -> None:
    """Update model population statistics.

    :param stat_book:
    :param gp_models:
    :return:
    """
    stat_book.update_stat_book(data=gp_models, x=x_train, base_kernels=base_kernel_names, n_dims=n_dims)


# stats functions
def get_model_scores(gp_models: List[GPModel], *args, **kwargs) -> List[float]:
    return [gp_model.score for gp_model in gp_models if gp_model.evaluated]


def get_n_operands(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    return [n_base_kernels(gp_model.covariance.raw_kernel) for gp_model in gp_models]


def get_n_hyperparams(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    return [gp_model.covariance.raw_kernel.size for gp_model in gp_models]


def get_cov_dists(gp_models: List[GPModel], *args, **kwargs) -> Union[np.ndarray, List[int]]:
    kernels = [gp_model.covariance for gp_model in gp_models]
    if len(kernels) >= 2:
        x = kwargs.get('x')
        return pairwise_centered_alignments(kernels, x)
    else:
        return [0] * len(gp_models)


def get_diversity_scores(gp_models: List[GPModel], *args, **kwargs) -> Union[float, List[int]]:
    kernels = [gp_model.covariance for gp_model in gp_models]
    base_kernels = kwargs.get('base_kernels')
    n_dims = kwargs.get('n_dims')
    return all_pairs_avg_dist(kernels, base_kernels, n_dims)


def get_best_n_operands(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    model_scores = get_model_scores(gp_models, *args, **kwargs)
    n_operands = get_n_operands(gp_models)
    score_arg_max = int(np.argmax(model_scores))
    return [n_operands[score_arg_max]]


def get_best_n_hyperparams(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    model_scores = get_model_scores(gp_models, *args, **kwargs)
    n_hyperparams = get_n_hyperparams(gp_models, *args, **kwargs)
    score_arg_max = int(np.argmax(model_scores))
    return [n_hyperparams[score_arg_max]]


def base_kern_freq(base_kern: str) -> Callable[[List[GPModel], Any, Any], List[int]]:
    def get_frequency(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
        cls = KERNEL_DICT[base_kern]
        return [count_kernel_types(gp_model.covariance.raw_kernel, lambda k: isinstance(k, cls)) for gp_model in
                gp_models]

    return get_frequency

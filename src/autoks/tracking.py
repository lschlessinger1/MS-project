from typing import List, Callable, Any, Union

import numpy as np

from src.autoks.backend.kernel import KERNEL_DICT, n_base_kernels
from src.autoks.core.covariance import all_pairs_avg_dist, pairwise_centered_alignments
from src.autoks.core.gp_model import GPModel
from src.autoks.core.model_selection import ModelSelector
from src.autoks.statistics import StatBook, StatBookCollection, Statistic
from src.autoks.util import type_count


class ModelSearchTracker:
    ### Should not have any dependency on grammar.
    # only depend on info from callback
    # think about stat book collection vs stat books
    # one tracker for each SB? or the collection
    # what about shared variables

    def __init__(self, base_kernel_names):
        # statistics used for plotting
        # base_kernel_names = base_kernel_names
        self.n_hyperparams_name = 'n_hyperparameters'
        self.n_operands_name = 'n_operands'
        self.base_kern_freq_names = [base_kern_name + '_frequency' for base_kern_name in base_kernel_names]
        self.score_name = 'score'
        self.cov_dists_name = 'cov_dists'
        self.diversity_scores_name = 'diversity_scores'
        self.best_stat_name = 'best'
        # All stat books track these variables
        shared_multi_stat_names = [self.n_hyperparams_name, self.n_operands_name] + self.base_kern_freq_names

        # raw value statistics
        base_kern_stat_funcs = [base_kern_freq(base_kern_name) for base_kern_name in base_kernel_names]
        shared_stats = [get_n_hyperparams, get_n_operands] + base_kern_stat_funcs

        # separate these!
        self.evaluations_name = 'evaluations'
        self.active_set_name = 'active_set'
        self.expansion_name = 'expansion'
        stat_book_names = [self.evaluations_name, self.expansion_name, self.active_set_name]
        self.stat_book_collection = StatBookCollection(stat_book_names, shared_multi_stat_names, shared_stats)

        sb_active_set = self.stat_book_collection.stat_books[self.active_set_name]
        sb_active_set.add_raw_value_stat(self.score_name, get_model_scores)
        sb_active_set.add_raw_value_stat(self.cov_dists_name, get_cov_dists)
        sb_active_set.add_raw_value_stat(self.diversity_scores_name, get_diversity_scores)
        sb_active_set.multi_stats[self.n_hyperparams_name].add_statistic(Statistic(self.best_stat_name,
                                                                                   get_best_n_hyperparams))
        sb_active_set.multi_stats[self.n_operands_name].add_statistic(Statistic(self.best_stat_name,
                                                                                get_best_n_operands))

        sb_evals = self.stat_book_collection.stat_books[self.evaluations_name]
        sb_evals.add_raw_value_stat(self.score_name, get_model_scores)

    def active_set_callback(self,
                            models: List[GPModel],
                            model_selector: ModelSelector,
                            x=None,
                            y=None) -> None:
        grammar = model_selector.grammar
        stat_book = self.stat_book_collection.stat_books[self.active_set_name]
        update_stat_book(stat_book, models, x, grammar.base_kernel_names, grammar.n_dims)

    def expansion_callback(self,
                           models: List[GPModel],
                           model_selector: ModelSelector,
                           x=None,
                           y=None) -> None:
        grammar = model_selector.grammar
        stat_book = self.stat_book_collection.stat_books[self.expansion_name]
        update_stat_book(stat_book, models, x, grammar.base_kernel_names, grammar.n_dims)

    def evaluations_callback(self,
                             models: List[GPModel],
                             model_selector: ModelSelector,
                             x=None,
                             y=None) -> None:
        grammar = model_selector.grammar
        stat_book = self.stat_book_collection.stat_books[self.evaluations_name]
        update_stat_book(stat_book, models, x, grammar.base_kernel_names, grammar.n_dims)


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
    if len(kernels) >= 2:
        base_kernels = kwargs.get('base_kernels')
        n_dims = kwargs.get('n_dims')
        return all_pairs_avg_dist(kernels, base_kernels, n_dims)
    else:
        return [0] * len(gp_models)


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
        return [type_count(gp_model.covariance.to_binary_tree(), cls) for gp_model in gp_models]

    return get_frequency
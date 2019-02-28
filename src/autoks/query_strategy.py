from abc import ABC
from typing import Callable, List

import numpy as np

from src.autoks.kernel import AKSKernel
from src.evalg.selection import Selector, AllSelector


# Query Strategies

class QueryStrategy(Selector, ABC):
    """Propose a model to evaluate

    Given a list of un-evaluated models, a scoring function, and training data
    we return x_star (chosen)
    """

    def __init__(self, n_individuals: int, scoring_func: Callable):
        super().__init__(n_individuals)
        self.scoring_func = scoring_func

    def query(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array):
        scores = self.score_kernels(kernels)
        ind = self.arg_select(np.array(kernels), scores)
        return ind, scores

    def score_kernels(self, kernels: List[AKSKernel]):
        return [self.scoring_func(kernel) for kernel in kernels]


class NaiveQueryStrategy(QueryStrategy, AllSelector):

    def __init__(self, n_individuals: int = 1, scoring_func: Callable = None):
        if scoring_func is None:
            scoring_func = score_all_same
        super().__init__(n_individuals, scoring_func)


class BestScoreStrategy(QueryStrategy):

    def __init__(self, scoring_func: Callable, n_individuals: int = 1):
        super().__init__(n_individuals, scoring_func)

    def select(self, population: np.array, scores: np.array):
        return self._select_helper(population, scores)

    def arg_select(self, population: np.array, scores: np.array):
        """Select best kernel according to scoring function

        :param population:
        :param scores:
        :return:
        """
        return int(np.argmax(scores))


# Scoring Functions

def score_all_same(kernel: AKSKernel):
    return 1


def expected_improvement(kernel: AKSKernel, hyperpriors, X_train: np.array, y_train: np.array):
    """Expected improvement (EI) acquisition function

    This acquisition function takes a model (kernel and hyperpriors) and computes expected improvement using
    the posterior Hellinger squared exponential covariance between models conditioned on the training data

    :param kernel:
    :return:
    """
    pass

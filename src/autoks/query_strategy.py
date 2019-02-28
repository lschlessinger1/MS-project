from typing import Callable, List

import numpy as np

from src.autoks.kernel import AKSKernel


# Query Strategies

class QueryStrategy:
    """Propose a model to evaluate

    Given a list of un-evaluated models, a scoring function, and training data
    we return x_star (chosen)
    """

    def __init__(self, scoring_func: Callable):
        self.scoring_func = scoring_func

    def select(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array):
        raise NotImplementedError('Method must be implemented in a child class')

    def score_kernels(self, kernels: List[AKSKernel]):
        return [self.scoring_func(kernel) for kernel in kernels]


class NaiveQueryStrategy(QueryStrategy):

    def __init__(self, scoring_func: Callable = None):
        if scoring_func is None:
            scoring_func = score_all_same
        super().__init__(scoring_func)

    def select(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array):
        """Select all kernels

        :param kernels:
        :param X_train:
        :param y_train:
        :return:
        """
        scores = self.score_kernels(kernels)
        return kernels, scores


class BestScoreStrategy(QueryStrategy):

    def __init__(self, scoring_func: Callable):
        super().__init__(scoring_func)

    def select(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array):
        """Select best kernel according to scoring func

        :param kernels:
        :param X_train:
        :param y_train:
        :return:
        """
        scores = self.score_kernels(kernels)
        x_star = kernels[int(np.argmax(scores))]
        return x_star, scores


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

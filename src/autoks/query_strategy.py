from abc import ABC
from typing import List, Tuple

import numpy as np

from src.autoks.kernel import AKSKernel
from src.evalg.selection import Selector, AllSelector


# Acquisition Functions

class AcquisitionFunction:

    def score(self, kernel: AKSKernel, X_train: np.array, y_train: np.array, hyperpriors=None) -> int:
        raise NotImplementedError('Must be implemented in a child class')


class UniformScorer(AcquisitionFunction):

    def score(self, kernel: AKSKernel, X_train: np.array, y_train: np.array, hyperpriors=None) -> int:
        return 1


class ExpectedImprovement(AcquisitionFunction):

    def score(self, kernel: AKSKernel, X_train: np.array, y_train: np.array, hyperpriors=None) -> int:
        """Expected improvement (EI) acquisition function

    This acquisition function takes a model (kernel and hyperpriors) and computes expected improvement using
    the posterior Hellinger squared exponential covariance between models conditioned on the training data

        :param kernel:
        :param X_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        pass


# Query Strategies

class QueryStrategy(Selector, ABC):
    """Propose a model to evaluate

    Given a list of un-evaluated models, a scoring function, and training data
    we return x_star (chosen)
    """

    def __init__(self, n_individuals: int, scoring_func: AcquisitionFunction):
        super().__init__(n_individuals)
        self.scoring_func = scoring_func

    def query(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array, hyperpriors=None) -> \
            Tuple[np.array, List[float]]:
        scores = self.score_kernels(kernels, X_train, y_train, hyperpriors)
        ind = self.arg_select(np.array(kernels), scores)
        return ind, scores

    def score_kernels(self, kernels: List[AKSKernel], X_train: np.array, y_train: np.array, hyperpriors=None) -> \
            List[float]:
        return [self.scoring_func.score(kernel, X_train, y_train, hyperpriors) for kernel in kernels]


class NaiveQueryStrategy(QueryStrategy, AllSelector):

    def __init__(self, n_individuals: int = 1, scoring_func: AcquisitionFunction = None):
        if scoring_func is None:
            scoring_func = UniformScorer()
        super().__init__(n_individuals, scoring_func)


class BestScoreStrategy(QueryStrategy):

    def __init__(self, scoring_func: AcquisitionFunction, n_individuals: int = 1):
        super().__init__(n_individuals, scoring_func)

    def select(self, population: np.array, scores: np.array) -> list:
        return self._select_helper(population, scores)

    def arg_select(self, population: np.array, scores: np.array) -> List[int]:
        """Select best kernel according to scoring function

        :param population:
        :param scores:
        :return:
        """
        return [int(np.argmax(scores))]

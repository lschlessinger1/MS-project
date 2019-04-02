from abc import ABC
from typing import List, Tuple, Optional

import numpy as np

from src.autoks.acquisition_function import AcquisitionFunction, UniformScorer
from src.autoks.hyperprior import Hyperpriors
from src.autoks.kernel import AKSKernel, get_kernel_mapping
from src.evalg.selection import Selector, AllSelector


class QueryStrategy(Selector, ABC):
    """Propose a model to evaluate

    Given a list of un-evaluated models, a scoring function, and training data
    we return x_star (chosen)
    """
    n_individuals: Optional[int]
    scoring_func: AcquisitionFunction

    def __init__(self, n_individuals, scoring_func):
        super().__init__(n_individuals)
        self.scoring_func = scoring_func

    def query(self,
              kernels: List[AKSKernel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None) -> Tuple[np.ndarray, List[float]]:
        """Query the next round of kernels using the acquisition function.

        :param kernels:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        if len(kernels) == 0:
            return np.array([]), []
        scores = self.score_kernels(kernels, x_train, y_train, hyperpriors)
        ind = self.arg_select(np.array(kernels), scores)
        return ind, scores

    def score_kernels(self,
                      kernels: List[AKSKernel],
                      x_train: np.ndarray,
                      y_train: np.ndarray,
                      hyperpriors: Optional[Hyperpriors] = None) -> List[float]:
        """Score all kernels using the scoring function.

        :param kernels:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return [self.scoring_func.score(kernel, x_train, y_train, hyperpriors) for kernel in kernels]


# Query Strategies


class BOMSInitQueryStrategy(QueryStrategy):

    def __init__(self, n_individuals=None, scoring_func=None):
        if scoring_func is None:
            scoring_func = UniformScorer()
        super().__init__(n_individuals, scoring_func)

    def arg_select(self, population: np.ndarray, fitness_list: np.ndarray) -> np.ndarray:
        """Select indices of all SE

        :param population:
        :param fitness_list:
        :return:
        """
        k_map = get_kernel_mapping()
        sq_exp_cls = k_map['SE']
        return np.array([i for (i, aks_kernel) in enumerate(population.tolist())
                         if isinstance(aks_kernel.kernel, sq_exp_cls)])


class NaiveQueryStrategy(QueryStrategy, AllSelector):
    scoring_func: Optional[AcquisitionFunction]

    def __init__(self, n_individuals=None, scoring_func=None):
        if scoring_func is None:
            scoring_func = UniformScorer()
        super().__init__(n_individuals, scoring_func)


class BestScoreStrategy(QueryStrategy):

    def __init__(self, scoring_func, n_individuals=1):
        super().__init__(n_individuals, scoring_func)

    def select(self,
               population: np.ndarray,
               scores: np.ndarray) -> np.ndarray:
        """See parent docstring.

        :param population:
        :param scores:
        :return:
        """
        return self._select(population, scores)

    def arg_select(self,
                   population: np.ndarray,
                   scores: np.ndarray) -> List[int]:
        """Select best kernel according to scoring function.

        :param population:
        :param scores:
        :return:
        """
        return [int(np.argmax(scores))]

from abc import ABC
from typing import List, Tuple, Optional

import numpy as np

from src.autoks.core.acquisition_function import AcquisitionFunction, UniformScorer
from src.autoks.core.gp_model import GPModel
from src.autoks.core.hyperprior import Hyperpriors
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
              unevaluated_kernels_ind: List[int],
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> Tuple[np.ndarray, List[float]]:
        """Query the next round of gp_models using the acquisition function.

        :param unevaluated_kernels_ind:
        :param all_kernels:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        candidates = [all_kernels[i] for i in unevaluated_kernels_ind]
        if len(candidates) == 0:
            return np.array([]), []
        scores = self.score_kernels(unevaluated_kernels_ind, all_kernels, x_train, y_train, hyperpriors,
                                    surrogate_model, **kwargs)
        ind = self.arg_select(np.array(candidates), np.array(scores))
        return ind, scores

    def score_kernels(self,
                      unevaluated_kernels_ind: List[int],
                      all_kernels: List[GPModel],
                      x_train: np.ndarray,
                      y_train: np.ndarray,
                      hyperpriors: Optional[Hyperpriors] = None,
                      surrogate_model: Optional = None,
                      **kwargs) -> List[float]:
        """Score all gp_models using the scoring function.

        :param kernels:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        return [self.scoring_func.score(ind, all_kernels, x_train, y_train, hyperpriors, surrogate_model, **kwargs)
                for ind in unevaluated_kernels_ind]

    def __repr__(self):
        return f'{self.__class__.__name__}('f'scoring_func={self.scoring_func!r})'

# Query Strategies


class BOMSInitQueryStrategy(QueryStrategy):

    def __init__(self, n_individuals=None, scoring_func=None):
        if scoring_func is None:
            scoring_func = UniformScorer()
        super().__init__(n_individuals, scoring_func)

    def arg_select(self, population: np.ndarray, fitness_list: np.ndarray) -> np.ndarray:
        """Select index of SE_0 + SE_0

        :param population:
        :param fitness_list:
        :return:
        """
        return np.array([0])


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
from typing import List

import numpy as np

from src.autoks.core.gp_model import GPModel
from src.evalg.selection import Selector, TruncationSelector, AllSelector, LinearRankingSelector


class KernelSelector:
    parent_selector: Selector
    offspring_selector: Selector
    kernel_pruner: Selector

    def __init__(self, parent_selector, offspring_selector, kernel_pruner):
        self.parent_selector = parent_selector
        self.offspring_selector = offspring_selector
        self.kernel_pruner = kernel_pruner

    @staticmethod
    def _select(kernels: List[GPModel],
                scores: List[float],
                selector: Selector) -> List[GPModel]:
        """Select a list of gp_models possibly using scores.

        :param kernels:
        :param scores:
        :param selector:
        :return:
        """
        return list(selector.select(np.array(kernels), np.array(scores)).tolist())

    def select_parents(self,
                       kernels: List[GPModel],
                       scores: List[float]) -> List[GPModel]:
        """Select parent gp_models.

        :param kernels:
        :param scores:
        :return:
        """
        return self._select(kernels, scores, self.parent_selector)

    def select_offspring(self,
                         kernels: List[GPModel],
                         scores: List[float]) -> List[GPModel]:
        """Select next round of gp_models.

        :param kernels:
        :param scores:
        :return:
        """
        return self._select(kernels, scores, self.offspring_selector)

    def prune_candidates(self,
                         kernels: List[GPModel],
                         scores: List[float]) -> List[GPModel]:
        """Remove candidates from kernel list.

        :param kernels:
        :param scores:
        :return:
        """
        return self._select(kernels, scores, self.kernel_pruner)


def BOMS_kernel_selector(n_parents: int = 1,
                         max_candidates: int = 600):
    """Construct a default BOMS kernel selector.

    :param n_parents: Number of parents to expand each round
    :param max_candidates: Max. number of un-evaluated models to keep each round
    :return:
    """
    parent_selector = TruncationSelector(n_parents)
    offspring_selector = AllSelector()
    kernel_pruner = TruncationSelector(max_candidates)
    return KernelSelector(parent_selector, offspring_selector, kernel_pruner)


def CKS_kernel_selector(n_parents: int = 1):
    """Construct a default CKS kernel selector.

    :param n_parents: Number of parents to expand each round
    :return:
    """
    parent_selector = TruncationSelector(n_parents)
    offspring_selector = AllSelector()
    kernel_pruner = AllSelector()
    return KernelSelector(parent_selector, offspring_selector, kernel_pruner)


def evolutionary_kernel_selector(n_parents: int = 1,
                                 max_offspring: int = 1000):
    """Construct a default evolutionary kernel selector.

    :param n_parents: Number of parents to expand each round
    :param max_offspring: Max. number of models to keep each round
    :return:
    """
    parent_selector = LinearRankingSelector(n_parents)
    offspring_selector = TruncationSelector(max_offspring)
    kernel_pruner = AllSelector()
    return KernelSelector(parent_selector, offspring_selector, kernel_pruner)

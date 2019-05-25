from typing import List

import numpy as np

from src.autoks.core.gp_model import GPModel
from src.autoks.core.kernel_encoding import KernelTree
from src.evalg.population import PopulationBase


class GPModelPopulation(PopulationBase):
    _models: List[GPModel]

    def __init__(self):
        self._models = []
        self._unique_model_hashes = set()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, new_models):
        self._models = []
        for model in new_models:
            self.add_model(model)

    def update(self, new_models: List[GPModel]):
        for new_model in new_models:
            if self._hash_model(new_model) not in self._unique_model_hashes:
                self.add_model(new_model)

    def add_model(self, model):
        self._models.append(model)
        self._unique_model_hashes.add(self._hash_model(model))

    @staticmethod
    def _hash_model(model):
        return model.covariance.symbolic_expr_expanded

    def genotypes(self) -> List[KernelTree]:
        return [gp_model.covariance.to_binary_tree() for gp_model in self.models]

    def phenotypes(self) -> List[GPModel]:
        return self.models

    def candidates(self):
        return [model for model in self.models if not model.evaluated]

    def variety(self) -> int:
        return len(self._unique_model_hashes)

    def size(self) -> int:
        return len(self)

    def sizes(self) -> List[int]:
        """Sizes of models in the population"""
        return [len(gene) for gene in self.genotypes()]

    def print_all(self) -> None:
        for model in self.models:
            model.covariance.pretty_print()

    def __len__(self):
        return len(self.models)


class ActiveModelPopulation(GPModelPopulation):
    """Evaluated models"""

    def objectives(self) -> List[float]:
        return [gp_model.score for gp_model in self.models]

    def avg_objective(self) -> float:
        """Average objective"""
        return float(np.mean(self.objectives()))

    def best_objective(self) -> float:
        """Objective of the highest scoring kernel"""
        return self.best_model().score

    def best_model(self) -> GPModel:
        """Model with the highest fitness score."""
        return self.models[int(np.argmax(self.objectives()))]

    def phenotypic_diversity(self) -> float:
        """Measure of the number of different solution behavior present."""
        raise NotImplementedError

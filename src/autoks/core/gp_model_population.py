from typing import List

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

    def scored_models(self):
        return [model for model in self.models if model.evaluated]

    def scores(self) -> List[float]:
        return [model.score for model in self.scored_models()]

    def variety(self) -> int:
        pass

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

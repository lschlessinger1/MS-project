from typing import List

from src.autoks.core.gp_model import GPModel
from src.autoks.core.kernel_encoding import KernelTree
from src.evalg.population import PopulationBase


class GPModelPopulation(PopulationBase):
    models: List[GPModel]

    def __init__(self):
        self.models = []

    def update(self, new_models: List[GPModel]):
        for new_model in new_models:
            self.models.append(new_model)

    def genotypes(self) -> List[KernelTree]:
        return [gp_model.covariance.to_binary_tree() for gp_model in self.models]

    def phenotypes(self) -> List[GPModel]:
        return self.models

    def candidates(self):
        return [model for model in self.models if not model.evaluated]

    def scored_models(self):
        return [model for model in self.models if model.evaluated]

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

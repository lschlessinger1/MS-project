from abc import ABC
from collections import Counter


class PopulationBase:

    def genotypes(self):
        """Get all genotypes in population"""
        raise NotImplementedError

    def phenotypes(self):
        """Get all phenotypes in population"""
        raise NotImplementedError

    def variety(self) -> int:
        """The number of distinct individuals in the population.

        :return:
        """
        raise NotImplementedError

    def size(self) -> int:
        """The number of individuals in the population (counting duplicates).

        :return:
        """
        raise NotImplementedError

    def print_all(self) -> None:
        """Print all individuals."""
        raise NotImplementedError


class Population(Counter, PopulationBase, ABC):
    """A multiset of genotypes.

    Individuals do not change.
    Keys are individuals and values are counts.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def variety(self) -> int:
        return len(self)

    def size(self) -> int:
        return sum(value for value in self.values())

    def print_all(self) -> None:
        for item in self:
            print(item)

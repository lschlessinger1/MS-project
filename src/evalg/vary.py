import importlib
from typing import List, Union, TypeVar

import numpy as np

from src.evalg.crossover import Recombinator
from src.evalg.mutation import Mutator
from src.evalg.serialization import Serializable


class Variator(Serializable):

    def __init__(self, operator: Union[Mutator, Recombinator]):
        self.operator = operator

    T = TypeVar('T')

    def vary(self, parents: List[T], **kwargs) -> List[T]:
        """Vary all parents.

        :param parents:
        :return:
        """
        raise NotImplementedError('vary must be implemented in a child class')

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["operator"] = self.operator.to_dict()
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        # Set operator
        operator_dict = input_dict["operator"]
        operator_class_name = operator_dict["__class__"]
        operator_module_name = operator_dict["__module__"]
        operator_module = importlib.import_module(operator_module_name)
        operator_class_ = getattr(operator_module, operator_class_name)
        input_dict["operator"] = operator_class_.from_dict(operator_dict)
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r})'


class CrossoverVariator(Variator):

    def __init__(self,
                 operator: Recombinator,
                 n_offspring: int,
                 n_way: int = 2,
                 c_prob: float = 0.9):
        """

        :param operator: the recombinator containing the crossover operator
        :param n_offspring: the number of individuals to return
        :param n_way: number of parents in crossover
        :param c_prob: probability of crossover
        """
        super().__init__(operator)
        self.n_offspring = n_offspring
        self.n_way = n_way
        self.c_prob = c_prob

    T = TypeVar('T')

    def crossover_all(self, parents: List[T]) -> List[T]:
        """Crossover applied to all parents.

        :param parents: the members of the population
        :return:
        """
        if len(parents) == 0:
            return []

        recombinator = self.operator

        offspring = []
        for i in range(0, self.n_offspring, self.n_way):
            if np.random.rand() < self.c_prob:
                selected_parents = [parents[(i + j) % len(parents)] for j in range(self.n_way)]
                children = recombinator.crossover(selected_parents)

                # add children to offspring
                for j, child in enumerate(children):
                    if len(offspring) < self.n_offspring:
                        offspring.append(child)

        return offspring

    def vary(self, parents: List[T], **kwargs) -> List[T]:
        """Crossover all parents.

        :param parents:
        :return:
        """
        return self.crossover_all(parents)

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["n_offspring"] = self.n_offspring
        input_dict["n_way"] = self.n_way
        input_dict["c_prob"] = self.c_prob
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, n_offspring={self.n_offspring!r}, ' \
            f'n_way={self.n_way!r}, c_prob={self.c_prob!r})'


class MutationVariator(Variator):

    def __init__(self,
                 operator: Mutator,
                 m_prob: float = 0.01):
        """

        :param operator: the mutator
        :param m_prob: probability of mutation
        """
        super().__init__(operator)
        self.m_prob = m_prob

    T = TypeVar('T')

    def mutate_all(self, individuals: List[T], **kwargs) -> List[T]:
        """Mutation applied to all offspring.

        :param individuals: the members of the population
        :return:
        """
        offspring = individuals.copy()
        mutator = self.operator

        offspring_mut = []

        for child in offspring:
            if np.random.rand() < self.m_prob:
                child = mutator.mutate(individual=child, **kwargs)

            offspring_mut.append(child)

        return offspring_mut

    def vary(self, parents: List[T], **kwargs) -> List[T]:
        return self.mutate_all(parents, **kwargs)

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["m_prob"] = self.m_prob
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, m_prob={self.m_prob!r})'


class PopulationOperator(Serializable):
    """Collection of variators."""

    def __init__(self, variators: List[Variator]):
        self._variators = variators

    @property
    def variators(self) -> List[Variator]:
        return self._variators

    @variators.setter
    def variators(self, variators: List[Variator]) -> None:
        if len(variators) == 0:
            raise ValueError('variators cannot be empty')
        if not all([isinstance(v, Variator) for v in variators]):
            raise TypeError(f'All items must be of type {Variator.__name__}')
        self._variators = variators

    T = TypeVar('T')

    def create_offspring(self, population: List[T], **kwargs) -> List[T]:
        """Create offspring by varying all population.

        :param population:
        :return:
        """
        offspring = population
        for variator in self.variators:
            offspring = variator.vary(offspring, **kwargs)
        return offspring

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["variators"] = [variator.to_dict() for variator in self.variators]
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict):
        input_dict = super()._format_input_dict(input_dict)
        input_dict["variators"] = [Variator.from_dict(v) for v in input_dict["variators"]]
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'variators={self.variators!r})'


class CrossMutPopOperator(PopulationOperator):
    """Perform both crossover then mutation to all individuals."""

    def __init__(self, variators: List[Union[CrossoverVariator, MutationVariator]]):
        super().__init__(variators)

    @property
    def variators(self) -> List[Union[CrossoverVariator, MutationVariator]]:
        return self._variators

    @variators.setter
    def variators(self, variators: List[Union[CrossoverVariator, MutationVariator]]) -> None:
        if len(variators) != 2:
            raise ValueError('Must have exactly 2 variators')
        self._variators = variators
        self._crossover_variator = self.variators[0]
        self._mutation_variator = self.variators[1]

    @property
    def crossover_variator(self) -> CrossoverVariator:
        return self._crossover_variator

    @crossover_variator.setter
    def crossover_variator(self, crossover_variator: CrossoverVariator) -> None:
        if not isinstance(crossover_variator, CrossoverVariator):
            raise TypeError('Variator must be of type %s' % CrossoverVariator.__name__)
        self._crossover_variator = crossover_variator

    @property
    def mutation_variator(self) -> MutationVariator:
        return self._mutation_variator

    @mutation_variator.setter
    def mutation_variator(self, mutation_variator: MutationVariator) -> None:
        if not isinstance(mutation_variator, MutationVariator):
            raise TypeError('Variator must be of type %s' % MutationVariator.__name__)
        self._mutation_variator = mutation_variator


class CrossoverPopOperator(PopulationOperator):

    def __init__(self, variators: List[CrossoverVariator]):
        super().__init__(variators)
        self._crossover_variator = self.variators[0]

    @property
    def variators(self) -> List[CrossoverVariator]:
        return self._variators

    @variators.setter
    def variators(self, variators: List[CrossoverVariator]) -> None:
        if len(self.variators) != 1:
            raise ValueError('Must have exactly 1 variator')
        self._variators = variators
        self._crossover_variator = self.variators[0]

    @property
    def crossover_variator(self) -> CrossoverVariator:
        return self._crossover_variator

    @crossover_variator.setter
    def crossover_variator(self, crossover_variator: CrossoverVariator) -> None:
        if not isinstance(crossover_variator, CrossoverVariator):
            raise TypeError('Variator must be of type %s' % CrossoverVariator.__name__)
        self._crossover_variator = crossover_variator


class MutationPopOperator(PopulationOperator):

    def __init__(self, variators: List[MutationVariator]):
        super().__init__(variators)
        self._mutation_variator = self.variators[0]

    @property
    def variators(self) -> List[MutationVariator]:
        return self._variators

    @variators.setter
    def variators(self, variators: List[MutationVariator]) -> None:
        if len(self.variators) != 1:
            raise ValueError('Must have exactly 1 variator')
        self._variators = variators
        self._mutation_variator = self.variators[0]

    @property
    def mutation_variator(self) -> MutationVariator:
        return self._mutation_variator

    @mutation_variator.setter
    def mutation_variator(self, mutation_variator: MutationVariator) -> None:
        if not isinstance(mutation_variator, MutationVariator):
            raise TypeError('First variator must be of type %s' % MutationVariator.__name__)
        self._mutation_variator = mutation_variator

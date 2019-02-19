from typing import List, Union

import numpy as np

from evalg.crossover import Recombinator
from evalg.mutation import Mutator


class Variator:

    def __init__(self, operator: Union[Mutator, Recombinator]):
        self.operator = operator

    def vary(self, parents: list):
        raise NotImplementedError('vary must be implemented in a child class')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r})'


class CrossoverVariator(Variator):

    def __init__(self, operator: Recombinator, n_offspring, n_way: int = 2, c_prob: float = 1.):
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

    def crossover_all(self, parents: list):
        """Crossover applied to all parents.

        :param parents: the members of the population
        :return:
        """
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

    def vary(self, parents: list):
        return self.crossover_all(parents)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, n_offspring={self.n_offspring!r}, ' \
            f'n_way={self.n_way!r}, c_prob={self.c_prob!r})'


class MutationVariator(Variator):

    def __init__(self, operator: Mutator, m_prob: float = 1.):
        """

        :param operator: the mutator
        :param m_prob: probability of mutation
        """
        super().__init__(operator)
        self.m_prob = m_prob

    def mutate_all(self, individuals: list):
        """Mutation applied to all offspring.

        :param individuals: the members of the population
        :return:
        """
        offspring = individuals.copy()
        mutator = self.operator

        offspring_mut = []

        for child in offspring:
            if np.random.rand() < self.m_prob:
                child = mutator.mutate(child)

            offspring_mut.append(child)

        return offspring_mut

    def vary(self, parents: list):
        return self.mutate_all(parents)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, m_prob={self.m_prob!r})'


class PopulationOperator:
    """ Collection of variators

    """

    def __init__(self, variators: List[Variator]):
        if len(variators) == 0:
            raise ValueError('variators cannot be empty')
        if not all([isinstance(v, Variator) for v in variators]):
            raise TypeError(f'All items must be of type {Variator.__name__}')
        self.variators = variators

    def create_offspring(self, population: list):
        offspring = population
        for variator in self.variators:
            offspring = variator.vary(offspring)
        return offspring


class CrossMutPopOperator(PopulationOperator):
    """Perform both crossover then mutation to all individuals"""

    def __init__(self, variators: List[Variator]):
        super().__init__(variators)
        if len(self.variators) != 2:
            raise ValueError('Must have exactly 2 variators')

        if not isinstance(self.variators[0], CrossoverVariator):
            raise TypeError('first variator must be of type %s' % CrossoverVariator.__name__)

        if not isinstance(self.variators[1], MutationVariator):
            raise TypeError('second variator must be of type %s' % MutationVariator.__name__)

        self.crossover_variator = self.variators[0]
        self.mutation_variator = self.variators[1]


class CrossoverPopOperator(PopulationOperator):

    def __init__(self, variators: List[Variator]):
        super().__init__(variators)
        if len(self.variators) != 1:
            raise ValueError('Must have exactly 1 variator')

        if not isinstance(self.variators[0], CrossoverVariator):
            raise TypeError('first variator must be of type %s' % CrossoverVariator.__name__)

        self.crossover_variator = self.variators[0]


class MutationPopOperator(PopulationOperator):

    def __init__(self, variators: List[Variator]):
        super().__init__(variators)
        if len(self.variators) != 1:
            raise ValueError('Must have exactly 1 variator')

        if not isinstance(self.variators[0], MutationVariator):
            raise TypeError('First variator must be of type %s' % MutationVariator.__name__)

        self.mutation_variator = self.variators[0]

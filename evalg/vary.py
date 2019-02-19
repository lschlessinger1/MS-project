import numpy as np

from evalg.crossover import Recombinator
from evalg.mutation import Mutator


class Variator:

    def __init__(self, operator):
        self.operator = operator

    def vary(self, parents):
        raise NotImplementedError('vary must be implemented in a child class')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r})'


class CrossoverVariator(Variator):

    def __init__(self, operator, n_offspring, n_way=2, c_prob=1.):
        """

        :param operator: the recombinator containing the crossover operator
        :param n_offspring: the number of individuals to return
        :param n_way: number of parents in crossover
        :param n_points: number of points in crossover operator
        :param c_prob: probability of crossover
        """
        super().__init__(operator)
        if not isinstance(operator, Recombinator):
            raise TypeError('operator must be of type 'f'{Recombinator.__name__}')
        self.n_offspring = n_offspring
        self.n_way = n_way
        self.c_prob = c_prob

    def crossover_all(self, parents):
        """Crossover applied to all parents.

        :param parents: the members of the population
        :return:
        """
        recombinator = self.operator

        offspring = []
        for i in range(0, self.n_offspring, self.n_way):
            if np.random.rand() < self.c_prob:
                selected_parents = [parents[(i + j) % len(parents)] for j in range(self.n_way)]

                recombinator.parents = selected_parents
                children = recombinator.crossover()

                # add children to offspring
                for j, child in enumerate(children):
                    if len(offspring) < self.n_offspring:
                        offspring.append(child)

        return offspring

    def vary(self, parents):
        return self.crossover_all(parents)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, n_offspring={self.n_offspring!r}, ' \
            f'n_way={self.n_way!r}, c_prob={self.c_prob!r})'


class MutationVariator(Variator):

    def __init__(self, operator, m_prob=1.):
        """

        :param operator: the mutator
        :param m_prob: probability of mutation
        """
        super().__init__(operator)
        if not isinstance(operator, Mutator):
            raise TypeError('operator must be of type %s' % Mutator.__name__)
        self.m_prob = m_prob

    def mutate_all(self, individuals):
        """Mutation applied to all offspring.

        :param individuals: the members of the population
        :return:
        """
        offspring = individuals.copy()
        mutator = self.operator

        offspring_mut = []

        for child in offspring:
            if np.random.rand() < self.m_prob:
                mutator.individual = child
                child = mutator.mutate()

            offspring_mut.append(child)

        return offspring_mut

    def vary(self, parents):
        return self.mutate_all(parents)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operator={self.operator!r}, m_prob={self.m_prob!r})'


class PopulationOperator:
    """ Collection of variators

    """

    def __init__(self, variators):
        # TODO make sure len > 0 and all of type Variator
        self.variators = variators

    def create_offspring(self, population):
        offspring = population
        for variator in self.variators:
            offspring = variator.vary(offspring)
        return offspring


def crossover_mutate_all(individuals, crossover_variator, mutation_variator):
    """Perform both crossover then mutation to all individuals

    :param individuals:
    :param crossover_variator:
    :param mutation_variator:
    :return:
    """
    if not isinstance(crossover_variator, CrossoverVariator):
        raise TypeError('crossover_variator must be of type %s' % CrossoverVariator.__name__)
    if not isinstance(mutation_variator, MutationVariator):
        raise TypeError('mutation_variator must be of type %s' % MutationVariator.__name__)

    offspring = crossover_variator.crossover_all(individuals)
    offspring = mutation_variator.mutate_all(offspring)
    return offspring

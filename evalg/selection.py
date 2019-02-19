from abc import ABC

import numpy as np


class Selector:

    def __init__(self, population, n_individuals):
        self.population = population

        if n_individuals < 0:
            raise ValueError('The number of individuals must be nonnegative.')
        self.n_individuals = n_individuals
        self.pop_size = self.population.shape[0]

    def select(self):
        # Select entire population if k > population size
        if self.n_individuals >= self.pop_size:
            return self.population

        ind = self.arg_select()
        individuals = self.population[ind]
        return individuals

    def arg_select(self):
        raise NotImplementedError("Implement arg_select in a child class")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class AllSelector(Selector):

    def __init__(self, population, n_individuals):
        super().__init__(population, n_individuals)

    def arg_select(self):
        return np.arange(self.pop_size)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class FitnessBasedSelector(Selector, ABC):

    def __init__(self, population, n_individuals, fitness_list):
        super().__init__(population, n_individuals)
        self.fitness_list = fitness_list

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class UniformSelector(Selector):

    def __init__(self, population, n_individuals):
        super().__init__(population, n_individuals)

    def arg_select(self):
        """Uniform Stochastic Selection

        Select the arguments of k individuals
        with replacement from the population uniformly
        at random.
        """
        ind = np.random.choice(self.pop_size, size=self.n_individuals, replace=True)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class StochasticUnivSampSelector(Selector):

    def __init__(self, population, n_individuals):
        super().__init__(population, n_individuals)

    def arg_select(self):
        """ Stochastic Universal Sampling
        """
        raise NotImplementedError("Stochastic Universal Sampling selection is not yet implemented.")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class BoltzmannSelector(Selector):

    def __init__(self, population, n_individuals, temperature, prev_pop_avg):
        super().__init__(population, n_individuals)
        self.temperature = temperature
        self.prev_pop_avg = prev_pop_avg

    def arg_select(self):
        """Boltzmann Selection
        """
        raise NotImplementedError("Boltzmann selection is not yet implemented.")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r}, ' \
            f'temperature={self.temperature!r}, prev_pop_avg={self.prev_pop_avg!r})'


class FitnessProportionalSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list):
        super().__init__(population, n_individuals, fitness_list)

    def arg_select(self):
        """Fitness-Proportional Selection

        Select k individuals with replacement from the population
        uniformly at random proportional to the fitness of
        the individual. This is also known as roulette-wheel
        selection or stochastic sampling with replacement.
        """
        probabilities = self.fitness_list / np.sum(self.fitness_list)
        ind = np.random.choice(self.pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class SigmaScalingSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list):
        super().__init__(population, n_individuals, fitness_list)

    def arg_select(self):
        """Sigma scaling selection
        """
        sigma = np.std(self.fitness_list)
        expected_cnts = np.empty(self.pop_size)
        if sigma > 0.0001:
            expected_cnts[:] = 1 + (self.fitness_list - np.mean(self.fitness_list)) / sigma
        else:
            expected_cnts[:] = 1

        max_exp_cnt = 1.5
        min_exp_cnt = 0

        expected_cnts[expected_cnts > max_exp_cnt] = max_exp_cnt
        expected_cnts[expected_cnts < min_exp_cnt] = min_exp_cnt

        probabilities = expected_cnts / np.sum(expected_cnts)
        ind = np.random.choice(self.pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class TruncationSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list):
        super().__init__(population, n_individuals, fitness_list)

    def arg_select(self):
        """Truncation selection

        Select k best from population according to fitness_list.

        k: number of individuals
        """
        ind = np.argpartition(self.fitness_list, -self.n_individuals)[-self.n_individuals:]
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class LinearRankingSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list):
        super().__init__(population, n_individuals, fitness_list)

    def arg_select(self):
        """Linear Ranking Selection

        Select k individuals with replacement from the population
        uniformly at random proportional to the relative fitness
        ranking of the individual.
        """
        rankings_asc = np.argsort(np.argsort(self.fitness_list)) + 1
        probabilities = rankings_asc / np.sum(rankings_asc)

        ind = np.random.choice(self.pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r})'


class ExponentialRankingSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list, c=0.99):
        super().__init__(population, n_individuals, fitness_list)

        if c <= 0 or c >= 1:
            raise ValueError("0 < c < 1 must hold")
        self.c = c

    def arg_select(self):
        """Exponential Ranking Selection
        """
        rankings_asc = np.argsort(np.argsort(self.fitness_list))
        probabilities = (self.c ** (self.pop_size - rankings_asc)) / np.sum(self.c ** (self.pop_size - rankings_asc))

        ind = np.random.choice(self.pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r}, ' \
            f'c={self.c!r})'


class TournamentSelector(FitnessBasedSelector):

    def __init__(self, population, n_individuals, fitness_list, n_way=2):
        super().__init__(population, n_individuals, fitness_list)

        if n_way < 2 or n_way > len(population):
            raise ValueError("The number of competitors in the tournament must be greater than 1 and less than the \
            number of individuals in the population.")
        self.n_way = n_way

    def arg_select(self):
        """Tournament Selection
        Uniformly at random select `n_way` individuals from the
        population then selecting the best (or worst) individual
        from the `n_way` competitors as the winner (or loser). There
        will be `k` tournaments run with replacement on the
        population.

        Notes
        -----
        Binary tournaments (`n_way` = 2) are equivalent to linear
        ranking selection in expectation. If `n_way` = 3, it is
        equivalent, in expectation, to quadratic ranking selection.
        """
        ind = np.empty(self.n_individuals)
        for i in range(self.n_individuals):
            selector = UniformSelector(self.population, self.n_way)
            rand_ind = selector.arg_select()
            winner = np.argmax(self.fitness_list[rand_ind], axis=0)
            ind[i] = winner

        return ind.astype(np.int)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, pop_size={self.pop_size!r}, ' \
            f'n_way={self.n_way!r})'

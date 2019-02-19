import numpy as np


class Selector:

    def __init__(self, n_individuals):
        if n_individuals < 0:
            raise ValueError('The number of individuals must be nonnegative.')
        self.n_individuals = n_individuals

    def _select_helper(self, population, fitness_list):
        # Select entire population if k > population size
        pop_size = population.shape[0]
        if self.n_individuals >= pop_size:
            return population

        return population[self.arg_select(population, fitness_list)]

    def select(self, population, fitness_list):
        raise NotImplementedError("Implement select in a child class")

    def arg_select(self, population, fitness_list):
        raise NotImplementedError("Implement arg_select in a child class")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class AllSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list=None):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list=None):
        return np.arange(population.shape[0])

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class UniformSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list=None):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list=None):
        """Uniform Stochastic Selection

        Select the arguments of k individuals
        with replacement from the population uniformly
        at random.
        """
        pop_size = population.shape[0]
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class StochasticUnivSampSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list=None):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list=None):
        """ Stochastic Universal Sampling
        """
        raise NotImplementedError("Stochastic Universal Sampling selection is not yet implemented.")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class BoltzmannSelector(Selector):

    def __init__(self, n_individuals, temperature, prev_pop_avg):
        super().__init__(n_individuals)
        self.temperature = temperature
        self.prev_pop_avg = prev_pop_avg

    def select(self, population, fitness_list=None):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list=None):
        """Boltzmann Selection
        """
        raise NotImplementedError("Boltzmann selection is not yet implemented.")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, temperature={self.temperature!r}, ' \
            f'prev_pop_avg={self.prev_pop_avg!r})'


class FitnessProportionalSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
        """Fitness-Proportional Selection

        Select k individuals with replacement from the population
        uniformly at random proportional to the fitness of
        the individual. This is also known as roulette-wheel
        selection or stochastic sampling with replacement.
        """
        pop_size = population.shape[0]
        probabilities = fitness_list / np.sum(fitness_list)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class SigmaScalingSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
        """Sigma scaling selection
        """
        pop_size = population.shape[0]
        sigma = np.std(fitness_list)
        expected_cnts = np.empty(pop_size)
        if sigma > 0.0001:
            expected_cnts[:] = 1 + (fitness_list - np.mean(fitness_list)) / sigma
        else:
            expected_cnts[:] = 1

        max_exp_cnt = 1.5
        min_exp_cnt = 0

        expected_cnts[expected_cnts > max_exp_cnt] = max_exp_cnt
        expected_cnts[expected_cnts < min_exp_cnt] = min_exp_cnt

        probabilities = expected_cnts / np.sum(expected_cnts)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class TruncationSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
        """Truncation selection

        Select k best from population according to fitness_list.

        k: number of individuals
        """
        ind = np.argpartition(fitness_list, -self.n_individuals)[-self.n_individuals:]
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class LinearRankingSelector(Selector):

    def __init__(self, n_individuals):
        super().__init__(n_individuals)

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
        """Linear Ranking Selection

        Select k individuals with replacement from the population
        uniformly at random proportional to the relative fitness
        ranking of the individual.
        """
        pop_size = population.shape[0]
        rankings_asc = np.argsort(np.argsort(fitness_list)) + 1
        probabilities = rankings_asc / np.sum(rankings_asc)

        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class ExponentialRankingSelector(Selector):

    def __init__(self, n_individuals, c=0.99):
        super().__init__(n_individuals)

        if c <= 0 or c >= 1:
            raise ValueError("0 < c < 1 must hold")
        self.c = c

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
        """Exponential Ranking Selection
        """
        pop_size = population.shape[0]
        rankings_asc = np.argsort(np.argsort(fitness_list))
        probabilities = (self.c ** (pop_size - rankings_asc)) / np.sum(self.c ** (pop_size - rankings_asc))

        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, c={self.c!r})'


class TournamentSelector(Selector):

    def __init__(self, n_individuals, n_way=2):
        super().__init__(n_individuals)

        if n_way < 2:
            raise ValueError("The number of competitors in the tournament must be greater than 1.")
        self.n_way = n_way

    def select(self, population, fitness_list):
        return self._select_helper(population, fitness_list)

    def arg_select(self, population, fitness_list):
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
        if self.n_way > len(population):
            raise ValueError("The number of competitors in the tournament must be less than the number of individuals "
                             "in the population.")

        ind = np.empty(self.n_individuals)
        for i in range(self.n_individuals):
            selector = UniformSelector(self.n_way)
            rand_ind = selector.arg_select(population)
            winner = np.argmax(fitness_list[rand_ind], axis=0)
            ind[i] = winner

        return ind.astype(np.int)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, n_way={self.n_way!r})'

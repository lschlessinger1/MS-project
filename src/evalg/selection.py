from typing import Optional

import numpy as np


class Selector:
    _n_individuals: Optional[int]

    def __init__(self, n_individuals: Optional[int] = None):
        self._n_individuals = n_individuals

    @property
    def n_individuals(self) -> Optional[int]:
        return self._n_individuals

    @n_individuals.setter
    def n_individuals(self, n_individuals: Optional[int]) -> None:
        if n_individuals is not None:
            if not isinstance(n_individuals, int):
                raise TypeError('The number of individuals must be an integer.')
            if n_individuals < 0:
                raise ValueError('The number of individuals must be non-negative.')
        self._n_individuals = n_individuals

    def _select(self,
                population: np.ndarray,
                fitness_list: Optional[np.ndarray]) -> np.ndarray:
        """Helper function to select from population using a fitness list.

        :param population:
        :param fitness_list:
        :return:
        """
        if fitness_list is not None and len(population) != len(fitness_list):
            raise ValueError('population and fitness list must have same shape')

        # Select entire population if k > population size
        pop_size = population.shape[0]
        if self.n_individuals is not None and self.n_individuals >= pop_size:
            return population

        # Treat NaNs as -inf.
        if fitness_list is not None:
            for i, score in enumerate(fitness_list):
                if np.isnan(score):
                    fitness_list[i] = -np.inf

        return population[self.arg_select(population, fitness_list)]

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Select from population.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Select indices from population.

        :param population:
        :param fitness_list:
        :return:
        """
        raise NotImplementedError("Implement arg_select in a child class")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r})'


class ProbabilityMixin:

    def get_probabilities(self, raw_fitness: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Implement get_probabilities in a child class')


class AllSelector(Selector):

    def __init__(self, n_individuals=None):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Select all.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Select all indices.

        :param population:
        :param fitness_list:
        :return:
        """
        return np.arange(population.shape[0])


class UniformSelector(Selector):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Uniform stochastic selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Uniform stochastic selection of indices.

        Select the arguments of k individuals
        with replacement from the population uniformly
        at random.
        """
        pop_size = population.shape[0]
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True)
        return ind


class StochasticUnivSampSelector(Selector):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Stochastic universal sampling selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Stochastic universal sampling selection of indices.

        :param population:
        :param fitness_list:
        :return:
        """
        raise NotImplementedError("Stochastic Universal Sampling selection is not yet implemented.")


class BoltzmannSelector(Selector):
    temperature: float
    prev_pop_avg: float

    def __init__(self, n_individuals: int, temperature, prev_pop_avg):
        super().__init__(n_individuals)
        self.temperature = temperature
        self.prev_pop_avg = prev_pop_avg

    def select(self,
               population: np.ndarray,
               fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Boltzmann selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: Optional[np.ndarray] = None) -> np.ndarray:
        """Boltzmann Selection of indices.

        :param population:
        :param fitness_list:
        :return:
        """
        raise NotImplementedError("Boltzmann selection is not yet implemented.")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, temperature={self.temperature!r},' \
            f' prev_pop_avg={self.prev_pop_avg!r})'


class FitnessProportionalSelector(Selector, ProbabilityMixin):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Fitness-proportional selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Fitness-proportional selection of indices.

        Select k individuals with replacement from the population
        uniformly at random proportional to the fitness of
        the individual. This is also known as roulette-wheel
        selection or stochastic sampling with replacement.
        """
        pop_size = population.shape[0]
        probabilities = self.get_probabilities(fitness_list)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def get_probabilities(self, raw_fitness: np.ndarray) -> np.ndarray:
        min_fitness = raw_fitness.min()
        if min_fitness < 0:
            fitness_normalized = raw_fitness + np.abs(min_fitness)
        else:
            fitness_normalized = raw_fitness
        return fitness_normalized / np.sum(fitness_normalized)


class SigmaScalingSelector(Selector, ProbabilityMixin):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Sigma scaling selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Sigma scaling selection of indices.

        :param population:
        :param fitness_list:
        :return:
        """
        pop_size = population.shape[0]
        probabilities = self.get_probabilities(fitness_list)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def get_probabilities(self, raw_fitness: np.ndarray) -> np.ndarray:
        pop_size = raw_fitness.shape[0]
        mu = np.mean(raw_fitness)
        sigma = np.std(raw_fitness)
        expected_cnts = np.empty(pop_size)
        if sigma > 0.0001:
            expected_cnts[:] = 1 + (raw_fitness - mu) / (2 * sigma)
        else:
            expected_cnts[:] = 1

        max_exp_cnt = 1.5
        min_exp_cnt = 0

        expected_cnts[expected_cnts > max_exp_cnt] = max_exp_cnt
        expected_cnts[expected_cnts < min_exp_cnt] = min_exp_cnt

        probabilities = expected_cnts / np.sum(expected_cnts)
        return probabilities


class TruncationSelector(Selector):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Truncation selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self, population: np.ndarray, fitness_list: np.ndarray) -> np.ndarray:
        """Truncation selection of indices.

        Select k best from population according to `fitness_list`.
        """
        ind = np.argpartition(fitness_list, -self.n_individuals)[-self.n_individuals:]
        return ind


class LinearRankingSelector(Selector, ProbabilityMixin):

    def __init__(self, n_individuals: int):
        super().__init__(n_individuals)

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Linear ranking selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Linear ranking selection of indices.

        Select k individuals with replacement from the population
        uniformly at random proportional to the relative fitness
        ranking of the individual.
        """
        pop_size = population.shape[0]
        probabilities = self.get_probabilities(fitness_list)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    @staticmethod
    def linear_rankings(fitness_list: np.ndarray) -> np.ndarray:
        """Best individual gets rank N and the worst one gets rank 1.

        :param fitness_list:
        :return:
        """
        return np.argsort(np.argsort(fitness_list)) + 1

    def get_probabilities(self, raw_fitness: np.ndarray) -> np.ndarray:
        rankings_asc = self.linear_rankings(raw_fitness)
        probabilities = rankings_asc / np.sum(rankings_asc)
        return probabilities


class ExponentialRankingSelector(Selector, ProbabilityMixin):
    _c: float

    def __init__(self,
                 n_individuals: int,
                 c: float = 0.99):
        super().__init__(n_individuals)
        self._c = c

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if c <= 0 or c >= 1:
            raise ValueError("0 < c < 1 must hold")
        self._c = c

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Exponential ranking selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Exponential ranking selection of indices.

        :param population:
        :param fitness_list:
        :return:
        """
        pop_size = population.shape[0]
        probabilities = self.get_probabilities(fitness_list)
        ind = np.random.choice(pop_size, size=self.n_individuals, replace=True, p=probabilities)
        return ind

    def get_probabilities(self, raw_fitness: np.ndarray) -> np.ndarray:
        pop_size = raw_fitness.shape[0]
        linear_rankings = LinearRankingSelector.linear_rankings(raw_fitness)
        probabilities = ((self.c - 1) / (self.c ** pop_size - 1)) * self.c ** (pop_size - linear_rankings)
        return probabilities

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_individuals={self.n_individuals!r}, c={self.c!r})'


class TournamentSelector(Selector):
    _n_way: int

    def __init__(self,
                 n_individuals: int,
                 n_way: int = 2):
        super().__init__(n_individuals)
        self._n_way = n_way

    @property
    def n_way(self) -> int:
        return self._n_way

    @n_way.setter
    def n_way(self, n_way: int) -> None:
        if n_way < 2:
            raise ValueError("The number of competitors in the tournament must be greater than 1.")
        self._n_way = n_way

    def select(self,
               population: np.ndarray,
               fitness_list: np.ndarray) -> np.ndarray:
        """Tournament selection.

        :param population:
        :param fitness_list:
        :return:
        """
        return self._select(population, fitness_list)

    def arg_select(self,
                   population: np.ndarray,
                   fitness_list: np.ndarray) -> np.ndarray:
        """Tournament selection of indices.

        Uniformly at random select `n_way` individuals from the
        population then selecting the best (or worst) individual
        from the `n_way` competitors as the winner (or loser). There
        will be `k` tournaments run with replacement on the
        population.

        Notes
        -----
        Unary tournaments (`n_way` = 1) are equivalent to random selection.
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

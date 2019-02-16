from unittest import TestCase

import numpy as np

from evalg.selection import Selector, UniformSelector, FitnessProportionalSelector, SigmaScalingSelector, \
    TruncationSelector, LinearRankingSelector, ExponentialRankingSelector, TournamentSelector


class TestSelector(TestCase):

    def setUp(self):
        self.population = np.array([1, 2, 3, 4, 5])

    def test_select(self):
        n = len(self.population)
        selector = Selector(self.population, n_individuals=n)
        result = selector.select()
        self.assertCountEqual(result.tolist(), self.population.tolist())

        n = len(self.population) + 1
        selector = Selector(self.population, n_individuals=n)
        result = selector.select()
        self.assertCountEqual(result.tolist(), self.population.tolist())

        n = min(1, len(self.population - 1))
        selector = Selector(self.population, n_individuals=n)

        with self.assertRaises(NotImplementedError):
            selector.select()
        with self.assertRaises(NotImplementedError):
            selector.arg_select()

        # Test negative k
        n = -1
        with self.assertRaises(ValueError):
            Selector(self.population, n_individuals=n)
        with self.assertRaises(ValueError):
            Selector(self.population, n_individuals=n)


class TestUniformSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])

    def test_select(self):
        selector = UniformSelector(self.population, n_individuals=3)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [4, 5, 3])

    def test_arg_select(self):
        selector = UniformSelector(self.population, n_individuals=3)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [3, 4, 2])


class TestFitnessProportionalSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = FitnessProportionalSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [2, 5, 4])

    def test_arg_select(self):
        selector = FitnessProportionalSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [1, 4, 3])


class TestSigmaScalingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = SigmaScalingSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [2, 4, 4])

    def test_arg_select(self):
        selector = SigmaScalingSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.arg_select()
        self.assertEqual(result.tolist(), [1, 3, 3])


class TestTruncationSelector(TestCase):

    def setUp(self):
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = TruncationSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [3, 4, 2])

    def test_arg_select(self):
        selector = TruncationSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [2, 3, 1])


class TestLinearRankingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = LinearRankingSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [2, 5, 4])

    def test_arg_select(self):
        selector = LinearRankingSelector(self.population, n_individuals=3, fitness_list=self.fitness_list)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [1, 4, 3])


class TestExponentialRankingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])
        self.n_indivs = 3
        self.c = 0.5

    def test_c_value(self):
        n = 1
        self.assertRaises(ValueError, ExponentialRankingSelector, self.population, n, self.fitness_list, c=2)
        self.assertRaises(ValueError, ExponentialRankingSelector, self.population, n, self.fitness_list, c=-1)

    def test_select(self):
        selector = ExponentialRankingSelector(self.population, self.n_indivs, self.fitness_list, self.c)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [3, 4, 4])

    def test_arg_select(self):
        selector = ExponentialRankingSelector(self.population, self.n_indivs, self.fitness_list, self.c)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [2, 3, 3])


class TestTournamentSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])
        self.n_indivs = 3
        self.n_way = 2

    def test_n_way_value(self):
        n = 1

        n_way = len(self.population) + 1
        self.assertRaises(ValueError, TournamentSelector, self.population, n, self.fitness_list, n_way=n_way)
        n_way = 1
        self.assertRaises(ValueError, TournamentSelector, self.population, n, self.fitness_list, n_way=n_way)

    def test_select(self):
        selector = TournamentSelector(self.population, self.n_indivs, self.fitness_list, self.n_way)
        result = selector.select()
        self.assertCountEqual(result.tolist(), [1, 1, 2])

    def test_arg_select(self):
        selector = TournamentSelector(self.population, self.n_indivs, self.fitness_list, self.n_way)
        result = selector.arg_select()
        self.assertCountEqual(result.tolist(), [0, 0, 1])
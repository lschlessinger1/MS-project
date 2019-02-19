from unittest import TestCase

import numpy as np

from evalg.selection import Selector, UniformSelector, FitnessProportionalSelector, SigmaScalingSelector, \
    TruncationSelector, LinearRankingSelector, ExponentialRankingSelector, TournamentSelector


class TestSelector(TestCase):

    def setUp(self):
        self.population = np.array([1, 2, 3, 4, 5])

    def test_select(self):
        # n = len(self.population)
        # selector = Selector(n_individuals=n)
        # result = selector.select(self.population, fitness_list=None)
        # self.assertCountEqual(result.tolist(), self.population.tolist())
        #
        # n = len(self.population) + 1
        # selector = Selector(n_individuals=n)
        # result = selector.select(self.population, fitness_list=None)
        # self.assertCountEqual(result.tolist(), self.population.tolist())

        n = min(1, len(self.population - 1))
        selector = Selector(n_individuals=n)
        self.assertRaises(NotImplementedError, selector.select, self.population, fitness_list=None)
        self.assertRaises(NotImplementedError, selector.arg_select, self.population, fitness_list=None)

        # Test negative n
        self.assertRaises(ValueError, Selector, n_individuals=-1)


class TestUniformSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])

    def test_select(self):
        selector = UniformSelector(n_individuals=3)
        result = selector.select(self.population)
        self.assertCountEqual(result.tolist(), [4, 5, 3])

    def test_arg_select(self):
        selector = UniformSelector(n_individuals=3)
        result = selector.arg_select(self.population)
        self.assertCountEqual(result.tolist(), [3, 4, 2])

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestFitnessProportionalSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = FitnessProportionalSelector(n_individuals=3)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [2, 5, 4])

    def test_arg_select(self):
        selector = FitnessProportionalSelector(n_individuals=3)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [1, 4, 3])

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestSigmaScalingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = SigmaScalingSelector(n_individuals=3)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [2, 4, 4])

    def test_arg_select(self):
        selector = SigmaScalingSelector(n_individuals=3)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertEqual(result.tolist(), [1, 3, 3])

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestTruncationSelector(TestCase):

    def setUp(self):
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = TruncationSelector(n_individuals=3)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [3, 4, 2])

    def test_arg_select(self):
        selector = TruncationSelector(n_individuals=3)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [2, 3, 1])


class TestLinearRankingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select(self):
        selector = LinearRankingSelector(n_individuals=3)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [2, 5, 4])

    def test_arg_select(self):
        selector = LinearRankingSelector(n_individuals=3)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [1, 4, 3])

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestExponentialRankingSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])
        self.n_indivs = 3
        self.c = 0.5

    def test_c_value(self):
        n = 1
        self.assertRaises(ValueError, ExponentialRankingSelector, n, c=2)
        self.assertRaises(ValueError, ExponentialRankingSelector, n, c=-1)

    def test_select(self):
        selector = ExponentialRankingSelector(self.n_indivs, self.c)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [3, 4, 4])

    def test_arg_select(self):
        selector = ExponentialRankingSelector(self.n_indivs, self.c)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [2, 3, 3])

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestTournamentSelector(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.population = np.array([1, 2, 3, 4, 5])
        self.fitness_list = np.array([15, 30, 20, 40, 10])
        self.n_indivs = 3
        self.n_way = 2

    def test_n_way_value(self):
        n = 1
        n_way = 1
        self.assertRaises(ValueError, TournamentSelector, n, n_way=n_way)

    def test_select(self):
        n_way = len(self.population) + 1
        selector = TournamentSelector(self.n_indivs, n_way)
        self.assertRaises(ValueError, selector.select, self.population, self.fitness_list)

        selector = TournamentSelector(self.n_indivs, self.n_way)
        result = selector.select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [1, 1, 2])

    def test_arg_select(self):
        n_way = len(self.population) + 1
        selector = TournamentSelector(self.n_indivs, n_way)
        self.assertRaises(ValueError, selector.arg_select, self.population, self.fitness_list)

        selector = TournamentSelector(self.n_indivs, self.n_way)
        result = selector.arg_select(self.population, self.fitness_list)
        self.assertCountEqual(result.tolist(), [0, 0, 1])

    def tearDown(self):
        # reset random seed
        np.random.seed()

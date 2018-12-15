from unittest import TestCase

import numpy as np

from evalg import selection


class TestSelection(TestCase):

    def setUp(self):
        self.population = np.array([1, 2, 3, 4, 5])
        self.k = 3
        self.fitness_list = np.array([15, 30, 20, 40, 10])

    def test_select_k_best(self):
        result = selection.select_k_best(self.population, self.fitness_list, self.k).tolist()
        self.assertEqual(result, [3, 4, 2])

    def test_select_exponential_ranking(self):
        with self.assertRaises(ValueError):
            selection.select_exponential_ranking(self.population, self.fitness_list, self.k, c=2)

        with self.assertRaises(ValueError):
            selection.select_exponential_ranking(self.population, self.fitness_list, self.k, c=-1)

    def test_select_tournament(self):
        with self.assertRaises(ValueError):
            selection.select_tournament(self.population, self.fitness_list, self.k, n_way=1)

        with self.assertRaises(ValueError):
            selection.select_tournament(self.population, self.fitness_list, self.k, n_way=len(self.population) + 1)

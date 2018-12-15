from unittest import TestCase

import numpy as np

from evalg import selection


class TestSelection(TestCase):

    def test_select_k_best(self):
        population = np.array([1, 2, 3, 4, 5])
        k = 3
        fitness_list = np.array([15, 30, 20, 40, 10])
        result = selection.select_k_best(population, fitness_list, k).tolist()
        self.assertEqual(result, [3, 4, 2])

    def test_select_exponential_ranking(self):
        population = np.array([1, 2])
        k = 2
        fitness_list = np.array([10, 15])
        with self.assertRaises(ValueError):
            selection.select_exponential_ranking(population, fitness_list, k, c=2)

        with self.assertRaises(ValueError):
            selection.select_exponential_ranking(population, fitness_list, k, c=-1)

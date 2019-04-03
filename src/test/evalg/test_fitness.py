from unittest import TestCase

from src.evalg.fitness import parsimony_pressure, covariant_parsimony_pressure


class TestFitness(TestCase):

    def test_parsimony_pressure(self):
        result = parsimony_pressure(fitness=8, size=10, p_coeff=0.2)
        self.assertIsInstance(result, float)
        self.assertEqual(6.0, result)

    def test_covariant_parsimony_pressure(self):
        fitness_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = covariant_parsimony_pressure(fitness=8, size=2, fitness_list=fitness_list, sizes=sizes)
        self.assertIsInstance(result, float)
        self.assertEqual(10.0, result)

from unittest import TestCase
from unittest.mock import MagicMock

from src.evalg.crossover import Recombinator
from src.evalg.mutation import Mutator
from src.evalg.vary import PopulationOperator, CrossMutPopOperator, CrossoverVariator, MutationVariator, \
    CrossoverPopOperator, MutationPopOperator


class TestPopulationOperator(TestCase):

    def test_variators(self):
        mock_variator = MagicMock()
        operator = PopulationOperator([mock_variator])
        with self.assertRaises(ValueError):
            operator.variators = []
        with self.assertRaises(TypeError):
            operator.variators = [3, 2, 3]


class TestCrossMutPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        operator = CrossMutPopOperator([cx_variator, mut_variator])
        with self.assertRaises(ValueError):
            operator.variators = [cx_variator, cx_variator, cx_variator]
        with self.assertRaises(TypeError):
            operator.mutation_variator = cx_variator
        with self.assertRaises(TypeError):
            operator.crossover_variator = mut_variator
        with self.assertRaises(ValueError):
            operator.variators = [mut_variator]


class TestCrossoverPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        operator = CrossoverPopOperator([cx_variator, mut_variator])
        with self.assertRaises(ValueError):
            operator.variators = [cx_variator, mut_variator]
        with self.assertRaises(TypeError):
            operator.crossover_variator = mut_variator


class TestMutationPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())

        operator = MutationPopOperator([cx_variator, mut_variator])
        with self.assertRaises(ValueError):
            operator.variators = [cx_variator, mut_variator]
        with self.assertRaises(TypeError):
            operator.mutation_variator = cx_variator

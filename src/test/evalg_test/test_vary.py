from unittest import TestCase

from src.evalg.crossover import Recombinator
from src.evalg.mutation import Mutator
from src.evalg.vary import PopulationOperator, CrossMutPopOperator, CrossoverVariator, MutationVariator, \
    CrossoverPopOperator, MutationPopOperator


class TestPopulationOperator(TestCase):

    def test_variators(self):
        self.assertRaises(ValueError, PopulationOperator, [])
        self.assertRaises(TypeError, PopulationOperator, [3, 2, 3])


class TestCrossMutPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        self.assertRaises(ValueError, CrossMutPopOperator, [cx_variator, cx_variator, cx_variator])
        self.assertRaises(TypeError, CrossMutPopOperator, [mut_variator, cx_variator])
        self.assertRaises(TypeError, CrossMutPopOperator, [mut_variator, mut_variator])
        self.assertRaises(ValueError, CrossMutPopOperator, [mut_variator])


class TestCrossoverPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        self.assertRaises(ValueError, CrossoverPopOperator, [cx_variator, mut_variator])
        self.assertRaises(TypeError, CrossoverPopOperator, [mut_variator])


class TestMutationPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        self.assertRaises(ValueError, MutationPopOperator, [cx_variator, mut_variator])
        self.assertRaises(TypeError, MutationPopOperator, [cx_variator])

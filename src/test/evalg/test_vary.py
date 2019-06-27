from unittest import TestCase
from unittest.mock import MagicMock

from src.evalg.crossover import Recombinator
from src.evalg.genprog import TreePointMutator, OnePointStrictLeafBiasedRecombinator
from src.evalg.mutation import Mutator
from src.evalg.vary import PopulationOperator, CrossMutPopOperator, CrossoverVariator, MutationVariator, \
    CrossoverPopOperator, MutationPopOperator, Variator


class TestVariator(TestCase):

    def test_to_dict(self):
        operator = TreePointMutator()
        variator = Variator(operator)
        result = variator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(operator.to_dict(), result["operator"])

    def test_from_dict(self):
        operator = TreePointMutator()
        variator = Variator(operator)
        result = Variator.from_dict(variator.to_dict())
        self.assertIsInstance(result, Variator)
        self.assertEqual(operator.binary_tree_node_cls, result.operator.binary_tree_node_cls)


class TestCrossoverVariator(TestCase):

    def test_to_dict(self):
        operator = OnePointStrictLeafBiasedRecombinator()
        variator = CrossoverVariator(operator, 10)
        result = variator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(operator.to_dict(), result["operator"])
        self.assertEqual(variator.n_offspring, result["n_offspring"])
        self.assertEqual(variator.n_way, result["n_way"])
        self.assertEqual(variator.c_prob, result["c_prob"])

    def test_from_dict(self):
        operator = OnePointStrictLeafBiasedRecombinator()
        variator = CrossoverVariator(operator, 3)
        result = CrossoverVariator.from_dict(variator.to_dict())
        self.assertIsInstance(result, CrossoverVariator)
        self.assertEqual(variator.operator.__class__, result.operator.__class__)
        self.assertEqual(variator.n_offspring, result.n_offspring)
        self.assertEqual(variator.n_way, result.n_way)
        self.assertEqual(variator.c_prob, result.c_prob)


class TestMutationVariator(TestCase):

    def test_to_dict(self):
        operator = TreePointMutator()
        variator = MutationVariator(operator)
        result = variator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(operator.to_dict(), result["operator"])
        self.assertEqual(variator.m_prob, result["m_prob"])

    def test_from_dict(self):
        operator = TreePointMutator()
        variator = MutationVariator(operator)
        result = CrossoverVariator.from_dict(variator.to_dict())
        self.assertIsInstance(result, MutationVariator)
        self.assertEqual(variator.operator.__class__, result.operator.__class__)
        self.assertEqual(variator.m_prob, result.m_prob)


class TestPopulationOperator(TestCase):

    def test_variators(self):
        mock_variator = MagicMock()
        operator = PopulationOperator([mock_variator])
        with self.assertRaises(ValueError):
            operator.variators = []
        with self.assertRaises(TypeError):
            operator.variators = [3, 2, 3]

    def test_to_dict(self):
        operator = TreePointMutator()
        variators = [MutationVariator(operator)]
        pop_operator = PopulationOperator(variators)
        result = pop_operator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual([v.to_dict() for v in variators], result["variators"])

    def test_from_dict(self):
        operator = TreePointMutator()
        variators = [MutationVariator(operator)]
        pop_operator = PopulationOperator(variators)
        result = PopulationOperator.from_dict(pop_operator.to_dict())
        self.assertIsInstance(result, PopulationOperator)
        self.assertEqual(pop_operator.variators[0].__class__, result.variators[0].__class__)


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

    def test_to_dict(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        pop_operator = CrossMutPopOperator([cx_variator, mut_variator])
        result = pop_operator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual([v.to_dict() for v in pop_operator.variators], result["variators"])

    def test_from_dict(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        pop_operator = CrossMutPopOperator([cx_variator, mut_variator])
        result = CrossMutPopOperator.from_dict(pop_operator.to_dict())
        self.assertIsInstance(result, CrossMutPopOperator)
        self.assertEqual(pop_operator.variators[0].__class__, result.variators[0].__class__)
        self.assertEqual(pop_operator.variators[1].__class__, result.variators[1].__class__)


class TestCrossoverPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())
        operator = CrossoverPopOperator([cx_variator, mut_variator])
        with self.assertRaises(ValueError):
            operator.variators = [cx_variator, mut_variator]
        with self.assertRaises(TypeError):
            operator.crossover_variator = mut_variator

    def test_to_dict(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        pop_operator = CrossoverPopOperator([cx_variator])
        result = pop_operator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual([v.to_dict() for v in pop_operator.variators], result["variators"])

    def test_from_dict(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        pop_operator = CrossoverPopOperator([cx_variator])
        result = CrossoverPopOperator.from_dict(pop_operator.to_dict())
        self.assertIsInstance(result, CrossoverPopOperator)
        self.assertEqual(pop_operator.variators[0].__class__, result.variators[0].__class__)


class TestMutationPopOperator(TestCase):

    def test_variators(self):
        cx_variator = CrossoverVariator(Recombinator(), 2)
        mut_variator = MutationVariator(Mutator())

        operator = MutationPopOperator([cx_variator, mut_variator])
        with self.assertRaises(ValueError):
            operator.variators = [cx_variator, mut_variator]
        with self.assertRaises(TypeError):
            operator.mutation_variator = cx_variator

    def test_to_dict(self):
        mut_variator = MutationVariator(Mutator())
        pop_operator = MutationPopOperator([mut_variator])
        result = pop_operator.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual([v.to_dict() for v in pop_operator.variators], result["variators"])

    def test_from_dict(self):
        mut_variator = MutationVariator(Mutator())
        pop_operator = MutationPopOperator([mut_variator])
        result = MutationPopOperator.from_dict(pop_operator.to_dict())
        self.assertIsInstance(result, MutationPopOperator)
        self.assertEqual(pop_operator.variators[0].__class__, result.variators[0].__class__)

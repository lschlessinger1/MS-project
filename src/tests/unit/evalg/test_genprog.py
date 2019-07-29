from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from src.evalg.crossover import Recombinator
from src.evalg.encoding import BinaryTree, BinaryTreeNode
from src.evalg.genprog import GrowGenerator, FullGenerator, TreePointMutator, FullMutator, HalfAndHalfMutator
from src.evalg.genprog.crossover import SubtreeExchangeRecombinatorBase, SubtreeExchangeRecombinator, \
    SubtreeExchangeLeafBiasedRecombinator, OnePointRecombinatorBase, OnePointRecombinator, \
    OnePointLeafBiasedRecombinator, OnePointStrictRecombinator
from src.evalg.genprog.generators import BinaryTreeGenerator, HalfAndHalfGenerator
from src.evalg.genprog.mutation import TreeMutator, SubTreeExchangeMutator, GrowMutator
from src.evalg.serialization import Serializable
from src.tests.unit.evalg.support.util import NodeCheckTestCase


class TestBinaryTreeGenerator(TestCase):

    def setUp(self):
        self.max_depth = 3
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.generator = BinaryTreeGenerator(self.max_depth, BinaryTreeNode)

    def test_max_depth(self):
        generator = BinaryTreeGenerator(max_depth=2, binary_tree_node_cls=BinaryTreeNode)
        with self.assertRaises(ValueError):
            generator.max_depth = -2

    def test_generate(self):
        self.assertRaises(NotImplementedError, self.generator.generate, self.operators, self.operands)

    def test_to_dict(self):
        actual = self.generator.to_dict()

        self.assertIsInstance(actual, dict)

        self.assertIn('max_depth', actual)
        self.assertIn('__module__', actual)
        self.assertIn('__class__', actual)
        self.assertIn('binary_tree_node_module_name', actual)
        self.assertIn('binary_tree_node_cls_name', actual)

        self.assertEqual(self.generator.max_depth, actual['max_depth'])
        self.assertEqual(self.generator.__module__, actual["__module__"])
        self.assertEqual(self.generator.__class__.__name__, actual["__class__"])
        self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual["binary_tree_node_module_name"])
        self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        test_cases = (BinaryTreeGenerator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                actual = cls.from_dict(self.generator.to_dict())

                self.assertIsInstance(actual, BinaryTreeGenerator)

                self.assertEqual(self.generator.max_depth, actual.max_depth)
                self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual.binary_tree_node_cls.__module__)
                self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual.binary_tree_node_cls.__name__)


class TestGrowGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = GrowGenerator(self.max_depth)

    def test_generate(self):
        tree = self.generator.generate(self.operators, self.operands, )
        self.assertIsInstance(tree, BinaryTree)
        self.assertLessEqual(tree.height(), self.max_depth + 1)  # depth of a stump is 0

    def test_grow(self):
        self.assertRaises(ValueError, self.generator.grow, self.operators, self.operands, -2)
        self.assertIsInstance(self.generator.grow(self.operators, self.operands, 0), BinaryTreeNode)

    def test_to_dict(self):
        actual = self.generator.to_dict()

        self.assertIsInstance(actual, dict)

        self.assertIn('max_depth', actual)
        self.assertIn('__module__', actual)
        self.assertIn('__class__', actual)
        self.assertIn('binary_tree_node_module_name', actual)
        self.assertIn('binary_tree_node_cls_name', actual)

        self.assertEqual(self.generator.max_depth, actual['max_depth'])
        self.assertEqual(self.generator.__module__, actual["__module__"])
        self.assertEqual(self.generator.__class__.__name__, actual["__class__"])
        self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual["binary_tree_node_module_name"])
        self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        test_cases = (GrowGenerator, BinaryTreeGenerator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                actual = cls.from_dict(self.generator.to_dict())

                self.assertIsInstance(actual, GrowGenerator)

                self.assertEqual(self.generator.max_depth, actual.max_depth)
                self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual.binary_tree_node_cls.__module__)
                self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual.binary_tree_node_cls.__name__)


class TestFullGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = FullGenerator(self.max_depth)

    def test_generate(self):
        tree = self.generator.generate(self.operators, self.operands, )
        self.assertIsInstance(tree, BinaryTree)
        max_height = self.max_depth + 1  # depth of a stump is 0
        self.assertLessEqual(tree.height(), max_height)

    def test_full(self):
        self.assertRaises(ValueError, self.generator.full, self.operators, self.operands, -2)
        self.assertIsInstance(self.generator.full(self.operators, self.operands, 0), BinaryTreeNode)

    def test_to_dict(self):
        actual = self.generator.to_dict()

        self.assertIsInstance(actual, dict)

        self.assertIn('max_depth', actual)
        self.assertIn('__module__', actual)
        self.assertIn('__class__', actual)
        self.assertIn('binary_tree_node_module_name', actual)
        self.assertIn('binary_tree_node_cls_name', actual)

        self.assertEqual(self.generator.max_depth, actual['max_depth'])
        self.assertEqual(self.generator.__module__, actual["__module__"])
        self.assertEqual(self.generator.__class__.__name__, actual["__class__"])
        self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual["binary_tree_node_module_name"])
        self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        test_cases = (FullGenerator, BinaryTreeGenerator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                actual = cls.from_dict(self.generator.to_dict())

                self.assertIsInstance(actual, FullGenerator)

                self.assertEqual(self.generator.max_depth, actual.max_depth)
                self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual.binary_tree_node_cls.__module__)
                self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual.binary_tree_node_cls.__name__)


class TestHalfAndHalfGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = HalfAndHalfGenerator(self.max_depth)

    def test_generate(self):
        tree = self.generator.generate(self.operators, self.operands)
        self.assertIsInstance(tree, BinaryTree)
        max_height = self.max_depth + 1  # depth of a stump is 0
        self.assertLessEqual(tree.height(), max_height)

    def test_to_dict(self):
        actual = self.generator.to_dict()

        self.assertIsInstance(actual, dict)

        self.assertIn('max_depth', actual)
        self.assertIn('__module__', actual)
        self.assertIn('__class__', actual)
        self.assertIn('binary_tree_node_module_name', actual)
        self.assertIn('binary_tree_node_cls_name', actual)

        self.assertEqual(self.generator.max_depth, actual['max_depth'])
        self.assertEqual(self.generator.__module__, actual["__module__"])
        self.assertEqual(self.generator.__class__.__name__, actual["__class__"])
        self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual["binary_tree_node_module_name"])
        self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        test_cases = (HalfAndHalfGenerator, BinaryTreeGenerator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                actual = cls.from_dict(self.generator.to_dict())

                self.assertIsInstance(actual, HalfAndHalfGenerator)

                self.assertEqual(self.generator.max_depth, actual.max_depth)
                self.assertEqual(self.generator.binary_tree_node_cls.__module__, actual.binary_tree_node_cls.__module__)
                self.assertEqual(self.generator.binary_tree_node_cls.__name__, actual.binary_tree_node_cls.__name__)


class TestTreeMutator(TestCase):

    def test_tree_type(self):
        self.assertRaises(TypeError, TreeMutator.mutate, 'bad type')
        self.assertRaises(TypeError, TreeMutator.mutate, 1)
        self.assertRaises(TypeError, TreeMutator.mutate, True)

    def test_to_dict(self):
        mutator = TreeMutator(BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("TreeMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        mutator = TreeMutator(BinaryTreeNode)
        actual = TreeMutator.from_dict(mutator.to_dict())
        self.assertIsInstance(actual, TreeMutator)
        self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)


class TestTreePointMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')
        np.random.seed(42)

    def test_mutate(self):
        mutator = TreePointMutator()
        tree = mutator.mutate(['+', '*'], ['A', 'B', 'C', 'D'], self.tree)
        self.assertEqual(tree.root.label, '+')
        self.assertIsInstance(tree, BinaryTree)

    def test_to_dict(self):
        mutator = TreePointMutator(BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("TreePointMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])

    def test_from_dict(self):
        test_cases = (TreePointMutator, TreeMutator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                mutator = TreePointMutator(BinaryTreeNode)
                actual = cls.from_dict(mutator.to_dict())
                self.assertIsInstance(actual, TreePointMutator)
                self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)

    def tearDown(self):
        # reset random seed
        np.random.seed()


class TestSubTreeExchangeMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')

    def test_max_depth(self):
        self.assertRaises(ValueError, SubTreeExchangeMutator, max_depth=-2, binary_tree_node_cls=BinaryTreeNode)

    def test__mutate_subtree_exchange(self):
        max_depth = 2
        tree_gen = GrowGenerator(max_depth)

        result = SubTreeExchangeMutator._mutate_subtree_exchange(['+', '*'], [1, 2, 3], self.tree, tree_gen)
        self.assertIsInstance(result, BinaryTree)
        max_height = max_depth + 1
        initial_height = self.tree.height()
        final_height = result.height()
        self.assertLessEqual(final_height, initial_height + max_height)

    def test__swap_mut_subtree(self):
        random_tree = BinaryTree()
        left = random_tree.root = BinaryTreeNode('*')
        ll = random_tree.root.add_left('C')
        lr = random_tree.root.add_right('D')

        r = 0  # A
        result = SubTreeExchangeMutator._swap_mut_subtree(self.tree, r, random_tree)
        self.assertIsInstance(result, BinaryTree)
        self.assertEqual(result.height(), 3)
        self.assertEqual(self.tree.root.left, left)
        self.assertEqual(self.tree.root.left.left, ll)
        self.assertEqual(self.tree.root.left.right, lr)

    def test_to_dict(self):
        mutator = SubTreeExchangeMutator(4, BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("SubTreeExchangeMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])
        self.assertEqual(mutator.max_depth, actual["max_depth"])

    def test_from_dict(self):
        test_cases = (SubTreeExchangeMutator, TreeMutator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                mutator = SubTreeExchangeMutator(4, BinaryTreeNode)
                actual = cls.from_dict(mutator.to_dict())
                self.assertIsInstance(actual, SubTreeExchangeMutator)
                self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)
                self.assertEqual(mutator.max_depth, actual.max_depth)


class TestGrowMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')

    def test_mutate(self):
        individual = self.tree
        operands = ['A', 'B', 'C']
        mutator = GrowMutator(max_depth=2)
        result = mutator.mutate(['+', '*'], operands, individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)

    def test_to_dict(self):
        mutator = GrowMutator(4, BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("GrowMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])
        self.assertEqual(mutator.max_depth, actual["max_depth"])

    def test_from_dict(self):
        test_cases = (GrowMutator, SubTreeExchangeMutator, TreeMutator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                mutator = GrowMutator(4, BinaryTreeNode)
                actual = cls.from_dict(mutator.to_dict())
                self.assertIsInstance(actual, GrowMutator)
                self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)
                self.assertEqual(mutator.max_depth, actual.max_depth)


class TestFullMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')

    def test_mutate(self):
        individual = self.tree
        operands = ['A', 'B', 'C']
        mutator = FullMutator(max_depth=2)
        result = mutator.mutate(['+', '*'], operands, individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)

    def test_to_dict(self):
        mutator = FullMutator(4, BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("FullMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])
        self.assertEqual(mutator.max_depth, actual["max_depth"])

    def test_from_dict(self):
        test_cases = (FullMutator, SubTreeExchangeMutator, TreeMutator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                mutator = FullMutator(4, BinaryTreeNode)
                actual = cls.from_dict(mutator.to_dict())
                self.assertIsInstance(actual, FullMutator)
                self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)
                self.assertEqual(mutator.max_depth, actual.max_depth)


class TestHalfAndHalfMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')

    def test_mutate(self):
        individual = self.tree
        operands = ['A', 'B', 'C']
        mutator = HalfAndHalfMutator(max_depth=2)
        result = mutator.mutate(['+', '*'], operands, individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)

    def test_to_dict(self):
        mutator = HalfAndHalfMutator(4, BinaryTreeNode)
        actual = mutator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.mutation", actual["__module__"])
        self.assertEqual("HalfAndHalfMutator", actual["__class__"])
        self.assertEqual("src.evalg.encoding", actual["binary_tree_node_module_name"])
        self.assertEqual("BinaryTreeNode", actual["binary_tree_node_cls_name"])
        self.assertEqual(mutator.max_depth, actual["max_depth"])

    def test_from_dict(self):
        test_cases = (HalfAndHalfMutator, SubTreeExchangeMutator, TreeMutator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                mutator = HalfAndHalfMutator(4, BinaryTreeNode)
                actual = cls.from_dict(mutator.to_dict())
                self.assertIsInstance(actual, HalfAndHalfMutator)
                self.assertEqual(BinaryTreeNode, actual.binary_tree_node_cls)
                self.assertEqual(mutator.max_depth, actual.max_depth)


class TestSubtreeExchangeRecombinatorBase(NodeCheckTestCase):

    def test_swap_same_node(self):
        node = BinaryTreeNode('*')
        tree = BinaryTree(node)
        a = b = node
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree, tree)

        root = tree.root
        self._check_node(root, '*', None, None, None)

    def test_swap_none_node(self):
        node = BinaryTreeNode('*')
        tree = BinaryTree(node)
        a = b = node
        SubtreeExchangeRecombinatorBase._swap_subtrees(None, b, tree, tree)

        root = tree.root
        self.check_stump(root, '*')
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, None, tree, tree)
        self.check_stump(root, '*')

    def test_swap_left_nodes(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1.left
        b = node_2.left

        # should be
        #     *         +
        #    / \       / \
        #   C   B  ,  A   D
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'C', 'B')
        self.check_leaf(root_1.left, 'C', '*')
        self.check_leaf(root_1.right, 'B', '*')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'A', 'D')
        self.check_leaf(root_2.left, 'A', '+')
        self.check_leaf(root_2.right, 'D', '+')

    def test_swap_right_nodes(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1.right
        b = node_2.right

        # should be
        #     *         +
        #    / \       / \
        #   A   D  ,  C   B
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'A', 'D')
        self.check_leaf(root_1.left, 'A', '*')
        self.check_leaf(root_1.right, 'D', '*')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'C', 'B')
        self.check_leaf(root_2.left, 'C', '+')
        self.check_leaf(root_2.right, 'B', '+')

    def test_swap_left_and_right_nodes(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1.left
        b = node_2.right

        # should be
        #     *         +
        #    / \       / \
        #   D   B  ,  C   A
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'D', 'B')
        self.check_leaf(root_1.left, 'D', '*')
        self.check_leaf(root_1.right, 'B', '*')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'C', 'A')
        self.check_leaf(root_2.left, 'C', '+')
        self.check_leaf(root_2.right, 'A', '+')

    def test_swap_right_and_left_nodes(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1.right
        b = node_2.left

        # should be
        #     *         +
        #    / \       / \
        #   A   C  ,  B   D
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'A', 'C')
        self.check_leaf(root_1.left, 'A', '*')
        self.check_leaf(root_1.right, 'C', '*')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'B', 'D')
        self.check_leaf(root_2.left, 'B', '+')
        self.check_leaf(root_2.right, 'D', '+')

    def test_swap_left_and_stump(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('C')
        tree_2 = BinaryTree(node_2)

        a = node_1.left
        b = node_2
        # should be
        #     *
        #    / \    ,  A
        #   C   B
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'C', 'B')
        self.check_leaf(root_1.left, 'C', '*')
        self.check_leaf(root_1.right, 'B', '*')
        root_2 = tree_2.root
        self.check_stump(root_2, 'A')

    def test_swap_right_and_stump(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('C')
        tree_2 = BinaryTree(node_2)

        a = node_1.right
        b = node_2
        # should be
        #     *
        #    / \    ,  B
        #   A   C
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'A', 'C')
        self.check_leaf(root_1.left, 'A', '*')
        self.check_leaf(root_1.right, 'C', '*')
        root_2 = tree_2.root
        self.check_stump(root_2, 'B')

    def test_swap_stump_and_node(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1.left
        b = node_2
        # should be
        #     *
        #    / \
        #   +   B   ,  A
        #  / \
        # C   D
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', '+', 'B')
        self._check_node(root_1.left, '+', 'C', 'D', '*')
        self.check_leaf(root_1.right, 'B', '*')
        self.check_leaf(root_1.left.left, 'C', '+')
        self.check_leaf(root_1.left.right, 'D', '+')

        root_2 = tree_2.root
        self.check_stump(root_2, 'A')

    def test_swap_nodes_with_children(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        tree_2 = BinaryTree(node_2)

        a = node_1
        b = node_2
        # should be
        #     +         *
        #    / \       / \
        #   C   D  ,  A   B
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'A', 'B')
        self.check_leaf(root_1.left, 'A', '*')
        self.check_leaf(root_1.right, 'B', '*')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'C', 'D')
        self.check_leaf(root_2.left, 'C', '+')
        self.check_leaf(root_2.right, 'D', '+')

    def test_swap_leaves(self):
        node_1 = BinaryTreeNode('A')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('B')
        tree_2 = BinaryTree(node_2)

        a = node_1
        b = node_2
        # should be
        #     A         B
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_stump(root_1, 'A')

        root_2 = tree_2.root
        self.check_stump(root_2, 'B')

    def test_swap_complex_trees(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        right = node_1.add_right('B')
        right.add_right('R')
        tree_1 = BinaryTree(node_1)

        node_2 = BinaryTreeNode('+')
        left = node_2.add_left('C')
        node_2.add_right('D')
        left.add_left('L')
        tree_2 = BinaryTree(node_2)

        a = node_1.right
        b = node_2.left
        # should be
        #     *           +
        #    / \         / \
        #   A   C   ,   B   D
        #      /         \
        #     L           R
        SubtreeExchangeRecombinatorBase._swap_subtrees(a, b, tree_1, tree_2)

        root_1 = tree_1.root
        self.check_root(root_1, '*', 'A', 'C')
        self.check_leaf(root_1.left, 'A', '*')
        self._check_node(root_1.right, 'C', 'L', None, '*')
        self.check_leaf(root_1.right.left, 'L', 'C')

        root_2 = tree_2.root
        self.check_root(root_2, '+', 'B', 'D')
        self._check_node(root_2.left, 'B', None, 'R', '+')
        self.check_leaf(root_2.right, 'D', '+')
        self.check_leaf(root_2.left.right, 'R', 'B')

    def test__valid_pair(self):
        result = SubtreeExchangeRecombinatorBase._valid_pair('A', 'B')
        self.assertTrue(result)
        result = SubtreeExchangeRecombinatorBase._valid_pair('+', '*')
        self.assertTrue(result)
        result = SubtreeExchangeRecombinatorBase._valid_pair('+', 'B')
        self.assertFalse(result)

    def test_to_dict(self):
        recombinator = SubtreeExchangeRecombinatorBase()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("SubtreeExchangeRecombinatorBase", actual["__class__"])

    def test_from_dict(self):
        test_cases = (SubtreeExchangeRecombinatorBase, Recombinator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = SubtreeExchangeRecombinatorBase()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, SubtreeExchangeRecombinatorBase)

    def tearDown(self):
        np.random.seed()


class TestSubtreeExchangeRecombinator(TestCase):

    def test_crossover(self):
        np.random.seed(10)
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_1.root.add_left('A')
        tree_1.root.add_right('B')

        tree_2 = BinaryTree(BinaryTreeNode('+'))
        tree_2.root.add_left('C')
        tree_2.root.add_right('D')

        # tests bad type
        self.assertRaises(TypeError, SubtreeExchangeRecombinator.crossover, 'bad type')
        self.assertRaises(TypeError, SubtreeExchangeRecombinator.crossover, [tree_1, tree_2, 45])

        parents = [tree_1, tree_2]
        recombinator = SubtreeExchangeRecombinator()
        result_1, result_2 = recombinator.crossover(parents)
        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.assertEqual(result_1.root.value, '*')
        self.assertEqual(result_1.root.left.value, 'A')
        self.assertEqual(result_1.root.right.value, 'D')
        self.assertEqual(result_2.root.value, '+')
        self.assertEqual(result_2.root.left.value, 'C')
        self.assertEqual(result_2.root.right.value, 'B')

    def test__select_token_ind(self):
        np.random.seed(10)
        tokens_1 = ['A', 'B', '+']
        tokens_2 = ['C', 'D', '*']
        idx_1, idx_2 = SubtreeExchangeRecombinator._select_token_ind(tokens_1, tokens_2)
        self.assertIsInstance(idx_1, int)
        self.assertIsInstance(idx_2, int)
        self.assertLess(idx_1, len(tokens_1))
        self.assertLess(idx_2, len(tokens_2))
        self.assertEqual(tokens_1[idx_1], 'B')
        self.assertEqual(tokens_2[idx_2], 'D')

    def test_to_dict(self):
        recombinator = SubtreeExchangeRecombinator()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("SubtreeExchangeRecombinator", actual["__class__"])

    def test_from_dict(self):
        test_cases = (SubtreeExchangeRecombinator, SubtreeExchangeRecombinatorBase, Recombinator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = SubtreeExchangeRecombinator()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, SubtreeExchangeRecombinator)

    def tearDown(self):
        np.random.seed()


class TestSubtreeExchangeLeafBiasedRecombinator(TestCase):

    def test_crossover(self):
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_1.root.add_left('A')
        tree_1.root.add_right('B')

        tree_2 = BinaryTree(BinaryTreeNode('+'))
        tree_2.root.add_left('C')
        tree_2.root.add_right('D')

        # tests bad type
        self.assertRaises(TypeError, SubtreeExchangeLeafBiasedRecombinator.crossover, 'bad type')
        self.assertRaises(TypeError, SubtreeExchangeLeafBiasedRecombinator.crossover, [tree_1, tree_2, 45])

        parents = [tree_1, tree_2]

        recombinator = SubtreeExchangeLeafBiasedRecombinator(t_prob=0)
        result_1, result_2 = recombinator.crossover(parents)
        self.assertEqual(result_1, tree_1)
        self.assertEqual(result_2, tree_2)

        recombinator = SubtreeExchangeLeafBiasedRecombinator(t_prob=1)
        result_1, result_2 = recombinator.crossover(parents)
        self.assertEqual(result_1.root.value, '*')
        self.assertEqual(result_2.root.value, '+')

        recombinator = SubtreeExchangeLeafBiasedRecombinator()
        result_1, result_2 = recombinator.crossover(parents)
        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        recombinator = SubtreeExchangeLeafBiasedRecombinator()
        stump = BinaryTree(BinaryTreeNode('C'))
        result_1, result_2 = recombinator.crossover([tree_1, stump])
        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

    def test_to_dict(self):
        recombinator = SubtreeExchangeLeafBiasedRecombinator(t_prob=0.3)
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("SubtreeExchangeLeafBiasedRecombinator", actual["__class__"])
        self.assertEqual(recombinator.t_prob, actual["t_prob"])

    def test_from_dict(self):
        test_cases = (SubtreeExchangeLeafBiasedRecombinator, SubtreeExchangeRecombinatorBase, Recombinator,
                      Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = SubtreeExchangeLeafBiasedRecombinator(t_prob=0.4)
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, SubtreeExchangeLeafBiasedRecombinator)
                self.assertEqual(recombinator.t_prob, actual.t_prob)

    def tearDown(self):
        np.random.seed()


class TestOnePointRecombinatorBase(NodeCheckTestCase):

    def setUp(self) -> None:
        self.recombinator = OnePointRecombinatorBase()

    def test_get_common_region(self):
        root_1 = BinaryTreeNode('*')
        root_1.add_left('B')
        right = root_1.add_right('+')
        right.add_left('D')
        rr = right.add_right('*')
        rr.add_left('F')
        rr.add_right('G')

        root_2 = BinaryTreeNode('+')
        left = root_2.add_left('+')
        right = root_2.add_right('*')
        left.add_left('K')
        left.add_right('L')
        right.add_right('M')
        right.add_left('N')

        result = self.recombinator.get_common_region(root_1, root_2)
        self.assertListEqual(result, [(root_1, root_2), (root_1.right, root_2.right),
                                      (root_1.right.left, root_2.right.left)])

    def test_select_node_pair(self):
        self.assertRaises(NotImplementedError, self.recombinator.select_node_pair, [])

    def test_crossover_stumps(self):
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_2 = BinaryTree(BinaryTreeNode('+'))

        parents = [tree_1, tree_2]

        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.assertEqual(result_1, tree_1)
        self.assertEqual(result_2, tree_2)

    def test_crossover_stump_and_tree(self):
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_1.root.add_left('A')
        tree_1.root.add_right('B')
        tree_2 = BinaryTree(BinaryTreeNode('+'))

        parents = [tree_1, tree_2]

        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.assertEqual(result_1, tree_1)
        self.assertEqual(result_2, tree_2)

    def test_crossover_roots(self):
        root = BinaryTreeNode('*')
        root.add_left('B')
        right = root.add_right('+')
        right.add_left('D')
        rr = right.add_right('*')
        rr.add_left('F')
        rr.add_right('G')
        tree_1 = BinaryTree(root)

        root = BinaryTreeNode('+')
        left = root.add_left('+')
        root.add_right('J')
        left.add_left('K')
        left.add_right('L')
        tree_2 = BinaryTree(root)

        parents = [tree_1, tree_2]

        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.assertEqual(result_1, tree_1)
        self.assertEqual(result_2, tree_2)

    def test_crossover_trees_roots_selected(self):
        root_1 = BinaryTreeNode('*')
        root_1.add_left('B')
        right = root_1.add_right('+')
        right.add_left('D')
        rr = right.add_right('*')
        rr.add_left('F')
        rr.add_right('G')
        tree_1 = BinaryTree(root_1)

        root_2 = BinaryTreeNode('+')
        left = root_2.add_left('+')
        right = root_2.add_right('*')
        left.add_left('K')
        left.add_right('L')
        right.add_right('M')
        right.add_left('N')
        tree_2 = BinaryTree(root_2)

        parents = [tree_1, tree_2]

        self.recombinator.select_node_pair = MagicMock()
        self.recombinator.select_node_pair.return_value = (root_1, root_2)
        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.recombinator.select_node_pair.assert_called_once()

        self.assertEqual(result_1, tree_1)
        self.assertEqual(result_2, tree_2)

    def test_crossover_subtrees(self):
        root_1 = BinaryTreeNode('*')
        root_1.add_left('B')
        right = root_1.add_right('+')
        right.add_left('D')
        rr = right.add_right('*')
        rr.add_left('F')
        rr.add_right('G')
        tree_1 = BinaryTree(root_1)

        root_2 = BinaryTreeNode('+')
        left = root_2.add_left('+')
        right = root_2.add_right('*')
        left.add_left('K')
        left.add_right('L')
        right.add_right('M')
        right.add_left('N')
        tree_2 = BinaryTree(root_2)

        parents = [tree_1, tree_2]

        self.recombinator.select_node_pair = MagicMock()
        self.recombinator.select_node_pair.return_value = (root_1.right, root_2.right)
        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.recombinator.select_node_pair.assert_called_once()

        self.check_root(result_1.root, '*', 'B', '*')
        self.check_leaf(result_1.root.left, 'B', '*')
        self._check_node(result_1.root.right, '*', 'N', 'M', '*')
        self.check_leaf(result_1.root.right.left, 'N', '*')
        self.check_leaf(result_1.root.right.right, 'M', '*')

        self.check_root(result_2.root, '+', '+', '+')
        self._check_node(result_2.root.left, '+', 'K', 'L', '+')
        self._check_node(result_2.root.right, '+', 'D', '*', '+')
        self.check_leaf(result_2.root.left.left, 'K', '+')
        self.check_leaf(result_2.root.left.right, 'L', '+')
        self.check_leaf(result_2.root.right.left, 'D', '+')
        self._check_node(result_2.root.right.right, '*', 'F', 'G', '+')
        self.check_leaf(result_2.root.right.right.left, 'F', '*')
        self.check_leaf(result_2.root.right.right.right, 'G', '*')

    def test_crossover_leaves(self):
        root_1 = BinaryTreeNode('*')
        root_1.add_left('B')
        right = root_1.add_right('+')
        right.add_left('D')
        rr = right.add_right('*')
        rr.add_left('F')
        rr.add_right('G')
        tree_1 = BinaryTree(root_1)

        root_2 = BinaryTreeNode('+')
        left = root_2.add_left('+')
        right = root_2.add_right('*')
        left.add_left('K')
        left.add_right('L')
        right.add_right('M')
        right.add_left('N')
        tree_2 = BinaryTree(root_2)

        parents = [tree_1, tree_2]

        self.recombinator.select_node_pair = MagicMock()
        self.recombinator.select_node_pair.return_value = (root_1.right.left, root_2.right.left)
        result_1, result_2 = self.recombinator.crossover(parents)

        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.recombinator.select_node_pair.assert_called_once()

        self.check_root(result_1.root, '*', 'B', '+')
        self.check_leaf(result_1.root.left, 'B', '*')
        self._check_node(result_1.root.right, '+', 'N', '*', '*')
        self.check_leaf(result_1.root.right.left, 'N', '+')
        self._check_node(result_1.root.right.right, '*', 'F', 'G', '+')
        self.check_leaf(result_1.root.right.right.left, 'F', '*')
        self.check_leaf(result_1.root.right.right.right, 'G', '*')

        self.check_root(result_2.root, '+', '+', '*')
        self._check_node(result_2.root.left, '+', 'K', 'L', '+')
        self._check_node(result_2.root.right, '*', 'D', 'M', '+')
        self.check_leaf(result_2.root.left.left, 'K', '+')
        self.check_leaf(result_2.root.left.right, 'L', '+')
        self.check_leaf(result_2.root.right.left, 'D', '*')
        self.check_leaf(result_2.root.right.right, 'M', '*')

    def test_to_dict(self):
        recombinator = OnePointRecombinatorBase()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("OnePointRecombinatorBase", actual["__class__"])

    def test_from_dict(self):
        test_cases = (OnePointRecombinatorBase, SubtreeExchangeRecombinatorBase, Recombinator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = OnePointRecombinatorBase()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, OnePointRecombinatorBase)

    def tearDown(self):
        np.random.seed()


class TestOnePointRecombinator(TestCase):

    def setUp(self) -> None:
        self.recombinator = OnePointRecombinator()

    def test_select_node_pair(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('E')
        node_4 = BinaryTreeNode('F')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIn(result, common_region)

    def test_to_dict(self):
        recombinator = OnePointRecombinator()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("OnePointRecombinator", actual["__class__"])

    def test_from_dict(self):
        test_cases = (OnePointRecombinator, OnePointRecombinatorBase, SubtreeExchangeRecombinatorBase, Recombinator,
                      Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = OnePointRecombinator()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, OnePointRecombinator)


class TestOnePointLeafBiasedRecombinator(NodeCheckTestCase):

    def setUp(self) -> None:
        self.recombinator = OnePointLeafBiasedRecombinator()

    def test_select_node_pair_one_pair(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        common_region = [(node_1, node_2)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertEqual(result, common_region[0])

    def test_select_node_pair_only_operators(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('+')
        node_3.add_left('A1')
        node_3.add_right('B1')
        node_4 = BinaryTreeNode('+')
        node_4.add_left('C1')
        node_4.add_right('D1')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIn(result, common_region)

    def test_select_node_pair_only_operands(self):
        node_1 = BinaryTreeNode('A')
        node_2 = BinaryTreeNode('B')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIn(result, common_region)

    def test_select_node_pair_operands_and_operators(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIn(result, common_region)

    def test_select_node_pair_t_prob_1(self):
        self.recombinator.t_prob = 1

        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertEqual(result, common_region[0])

    def test_select_node_pair_t_prob_0(self):
        self.recombinator.t_prob = 0

        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertEqual(result, common_region[1])

    def test_to_dict(self):
        recombinator = OnePointLeafBiasedRecombinator()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("OnePointLeafBiasedRecombinator", actual["__class__"])

    def test_from_dict(self):
        test_cases = (OnePointLeafBiasedRecombinator, OnePointRecombinatorBase, SubtreeExchangeRecombinatorBase,
                      Recombinator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = OnePointLeafBiasedRecombinator()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, OnePointLeafBiasedRecombinator)


class TestOnePointStrictRecombinator(NodeCheckTestCase):

    def setUp(self) -> None:
        self.recombinator = OnePointStrictRecombinator()

    def test_select_node_pair_one_pair(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')
        common_region = [(node_1, node_2)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIsNone(result)

    def test_select_node_pair_dif_operator(self):
        node_1 = BinaryTreeNode('+')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('*')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertEqual(common_region[1], result)

    def test_select_node_pair_same_operator(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')
        node_2 = BinaryTreeNode('*')
        node_2.add_left('C')
        node_2.add_right('D')
        node_3 = BinaryTreeNode('C')
        node_4 = BinaryTreeNode('D')
        common_region = [(node_1, node_2), (node_3, node_4)]
        result = self.recombinator.select_node_pair(common_region)
        self.assertIn(result, common_region)

    def test_to_dict(self):
        recombinator = OnePointStrictRecombinator()
        actual = recombinator.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertEqual("src.evalg.genprog.crossover", actual["__module__"])
        self.assertEqual("OnePointStrictRecombinator", actual["__class__"])

    def test_from_dict(self):
        test_cases = (OnePointStrictRecombinator, OnePointRecombinatorBase, SubtreeExchangeRecombinatorBase,
                      Recombinator, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                recombinator = OnePointStrictRecombinator()
                actual = cls.from_dict(recombinator.to_dict())
                self.assertIsInstance(actual, OnePointStrictRecombinator)

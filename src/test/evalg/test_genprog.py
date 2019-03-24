from unittest import TestCase

import numpy as np

from src.evalg.encoding import BinaryTree, BinaryTreeNode
from src.evalg.genprog import TreePointMutator, TreeMutator, SubTreeExchangeMutator, BinaryTreeGenerator, \
    GrowGenerator, FullGenerator, GrowMutator, FullMutator, SubtreeExchangeBinaryRecombinator, HalfAndHalfMutator, \
    HalfAndHalfGenerator, OnePointRecombinator


class TestBinaryTreeGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 3
        self.generator = BinaryTreeGenerator(self.operators, self.operands, self.max_depth)

    def test_max_depth(self):
        generator = BinaryTreeGenerator(self.operators, self.operands, max_depth=2)
        with self.assertRaises(ValueError):
            generator.max_depth = -2

    def test_generate(self):
        self.assertRaises(NotImplementedError, self.generator.generate)


class TestGrowGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = GrowGenerator(self.operators, self.operands, self.max_depth)

    def test_generate(self):
        tree = self.generator.generate()
        self.assertIsInstance(tree, BinaryTree)
        self.assertLessEqual(tree.height(), self.max_depth + 1)  # depth of a stump is 0

    def test_grow(self):
        self.assertRaises(ValueError, self.generator.grow, -2)
        self.assertIsInstance(self.generator.grow(0), BinaryTreeNode)


class TestFullGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = FullGenerator(self.operators, self.operands, self.max_depth)

    def test_generate(self):
        tree = self.generator.generate()
        self.assertIsInstance(tree, BinaryTree)
        max_height = self.max_depth + 1  # depth of a stump is 0
        self.assertLessEqual(tree.height(), max_height)

    def test_full(self):
        self.assertRaises(ValueError, self.generator.full, -2)
        self.assertIsInstance(self.generator.full(0), BinaryTreeNode)


class TestHalfAndHalfGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 2
        self.generator = HalfAndHalfGenerator(self.operators, self.operands, self.max_depth)

    def test_generate(self):
        tree = self.generator.generate()
        self.assertIsInstance(tree, BinaryTree)
        max_height = self.max_depth + 1  # depth of a stump is 0
        self.assertLessEqual(tree.height(), max_height)


class TestTreeMutator(TestCase):

    def test_tree_type(self):
        self.assertRaises(TypeError, TreeMutator.mutate, 'bad type')
        self.assertRaises(TypeError, TreeMutator.mutate, 1)
        self.assertRaises(TypeError, TreeMutator.mutate, True)


class TestTreePointMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')
        np.random.seed(42)

    def test_mutate(self):
        mutator = TreePointMutator(operands=['A', 'B', 'C', 'D'])
        tree = mutator.mutate(self.tree)
        self.assertEqual(tree.root.label, '+')
        self.assertIsInstance(tree, BinaryTree)

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
        operands = ['A', 'B', 'C', 'D']
        self.assertRaises(ValueError, SubTreeExchangeMutator, operands, max_depth=-2)

    def test__mutate_subtree_exchange(self):
        max_depth = 2
        tree_gen = GrowGenerator(['+', '*'], [1, 2, 3], max_depth)

        result = SubTreeExchangeMutator._mutate_subtree_exchange(self.tree, tree_gen)
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
        mutator = GrowMutator(operands, max_depth=2)
        result = mutator.mutate(individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)


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
        mutator = FullMutator(operands, max_depth=2)
        result = mutator.mutate(individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)


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
        mutator = HalfAndHalfMutator(operands, max_depth=2)
        result = mutator.mutate(individual)
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)


class TestSubtreeExchangeBinaryRecombinator(TestCase):

    def test__swap_subtrees(self):
        node_1 = BinaryTreeNode('*')
        node_1.add_left('A')
        node_1.add_right('B')

        node_2 = BinaryTreeNode('+')
        node_2.add_left('C')
        node_2.add_right('D')

        new_node_1, new_node_2 = SubtreeExchangeBinaryRecombinator._swap_subtrees(node_1.left, node_2.left)

        self.assertIsInstance(new_node_1, BinaryTreeNode)
        self.assertIsInstance(new_node_2, BinaryTreeNode)

        self.assertEqual(new_node_1.value, 'A')
        self.assertEqual(new_node_1.parent.value, '+')
        self.assertEqual(new_node_1.parent.left.value, 'A')
        self.assertEqual(new_node_1.parent.right.value, 'D')
        self.assertEqual(new_node_2.value, 'C')
        self.assertEqual(new_node_2.parent.value, '*')
        self.assertEqual(new_node_2.parent.left.value, 'C')
        self.assertEqual(new_node_2.parent.right.value, 'B')

    def test__valid_pair(self):
        result = SubtreeExchangeBinaryRecombinator._valid_pair('A', 'B')
        self.assertTrue(result)
        result = SubtreeExchangeBinaryRecombinator._valid_pair('+', '*')
        self.assertTrue(result)
        result = SubtreeExchangeBinaryRecombinator._valid_pair('+', 'B')
        self.assertFalse(result)

    def test__select_token_ind(self):
        np.random.seed(10)
        tokens_1 = ['A', 'B', '+']
        tokens_2 = ['C', 'D', '*']
        idx_1, idx_2 = SubtreeExchangeBinaryRecombinator._select_token_ind(tokens_1, tokens_2)
        self.assertIsInstance(idx_1, int)
        self.assertIsInstance(idx_2, int)
        self.assertLess(idx_1, len(tokens_1))
        self.assertLess(idx_2, len(tokens_2))
        self.assertEqual(tokens_1[idx_1], 'B')
        self.assertEqual(tokens_2[idx_2], 'D')

    def tearDown(self):
        np.random.seed()


class TestOnePointRecombinator(TestCase):

    def test_crossover(self):
        np.random.seed(10)
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_1.root.add_left('A')
        tree_1.root.add_right('B')

        tree_2 = BinaryTree(BinaryTreeNode('+'))
        tree_2.root.add_left('C')
        tree_2.root.add_right('D')

        # test bad type
        self.assertRaises(TypeError, OnePointRecombinator.crossover, 'bad type')
        self.assertRaises(TypeError, OnePointRecombinator.crossover, [tree_1, tree_2, 45])

        parents = [tree_1, tree_2]
        recombinator = OnePointRecombinator()
        result_1, result_2 = recombinator.crossover(parents)
        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        self.assertEqual(result_1.root.value, '*')
        self.assertEqual(result_1.root.left.value, 'A')
        self.assertEqual(result_1.root.right.value, 'D')
        self.assertEqual(result_2.root.value, '+')
        self.assertEqual(result_2.root.left.value, 'C')
        self.assertEqual(result_2.root.right.value, 'B')

    def tearDown(self):
        np.random.seed()

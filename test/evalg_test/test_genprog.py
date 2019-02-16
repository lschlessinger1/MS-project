from unittest import TestCase

import numpy as np

from evalg.encoding import BinaryTree, BinaryTreeNode
from evalg.genprog import TreePointMutator, TreeMutator, TreeRecombinator, BinaryTreeRecombinator, \
    SubTreeExchangeMutator, BinaryTreeGenerator, GrowGenerator, FullGenerator, GrowMutator, HalfAndHalfMutator, \
    SubtreeExchangeBinaryRecombinator


class TestBinaryTreeGenerator(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.operands = [1, 2, 3]
        self.max_depth = 3
        self.generator = BinaryTreeGenerator(self.operators, self.operands, self.max_depth)

    def test_max_depth(self):
        self.assertRaises(ValueError, BinaryTreeGenerator, self.operators, self.operands, max_depth=-2)

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


class TestTreeMutator(TestCase):

    def test_tree_type(self):
        self.assertRaises(TypeError, TreeMutator, 'bad type')
        self.assertRaises(TypeError, TreeMutator, 1)
        self.assertRaises(TypeError, TreeMutator, True)


class TestTreeRecombinator(TestCase):

    def test_tree_type(self):
        tree_1 = BinaryTree()
        tree_2 = BinaryTree()
        self.assertRaises(TypeError, TreeRecombinator, 'bad type')
        self.assertRaises(TypeError, TreeRecombinator, [tree_1, tree_2, 45])


class TestBinaryTreeRecombinator(TestCase):

    def test_tree_type(self):
        tree_1 = BinaryTree()
        tree_2 = BinaryTree()
        recombinator = BinaryTreeRecombinator([tree_1, tree_2])
        self.assertIsInstance(recombinator.parent_1, BinaryTree)
        self.assertIsInstance(recombinator.parent_2, BinaryTree)


class TestTreePointMutator(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root
        self.root.add_left('A')
        self.root.add_right('B')
        np.random.seed(42)

    def test_mutate(self):
        mutator = TreePointMutator(individual=self.tree, operands=['A', 'B', 'C', 'D'])
        tree = mutator.mutate()
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
        self.assertRaises(ValueError, SubTreeExchangeMutator, self.tree, operands, max_depth=-2)

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
        l = random_tree.root = BinaryTreeNode('*')
        ll = random_tree.root.add_left('C')
        lr = random_tree.root.add_right('D')

        r = 0  # A
        result = SubTreeExchangeMutator._swap_mut_subtree(self.tree, r, random_tree)
        self.assertIsInstance(result, BinaryTree)
        self.assertEqual(result.height(), 3)
        self.assertEqual(self.tree.root.left, l)
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
        mutator = GrowMutator(individual, operands, max_depth=2)
        result = mutator.mutate()
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
        mutator = HalfAndHalfMutator(individual, operands, max_depth=2)
        result = mutator.mutate()
        self.assertIsInstance(result, BinaryTree)
        max_height = mutator.max_depth + 1
        self.assertLessEqual(result.height(), self.tree.height() + max_height)


class TestSubtreeExchangeBinaryRecombinator(TestCase):

    def setUp(self):
        pass

    def test_crossover(self):
        tree_1 = BinaryTree()
        tree_1.root = BinaryTreeNode('*')
        tree_1.root = tree_1.root
        tree_1.root.add_left('A')
        tree_1.root.add_right('B')

        tree_2 = BinaryTree()
        tree_2.root = BinaryTreeNode('+')
        tree_2.root = tree_1.root
        tree_2.root.add_left('C')
        tree_2.root.add_left('D')

        parents = [tree_1, tree_2]
        recombinator = SubtreeExchangeBinaryRecombinator(parents)
        result_1, result_2 = recombinator.crossover()
        self.assertIsInstance(result_1, BinaryTree)
        self.assertIsInstance(result_2, BinaryTree)

        # TODO: test correctness of results

    def test__swap_subtrees(self):
        # TODO: test correctness of SubtreeExchangeBinaryRecombinator._swap_subtrees
        pass

    def test__valid_pair(self):
        # TODO: test correctness of SubtreeExchangeBinaryRecombinator._valid_pair
        pass

    def test__select_token_ind(self):
        # TODO: test correctness of SubtreeExchangeBinaryRecombinator._select_token_ind
        pass
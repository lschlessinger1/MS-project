from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from src.evalg.encoding import BinaryTree, BinaryTreeNode
from src.evalg.genprog import TreePointMutator, TreeMutator, SubTreeExchangeMutator, BinaryTreeGenerator, \
    GrowGenerator, FullGenerator, GrowMutator, FullMutator, SubtreeExchangeRecombinatorBase, HalfAndHalfMutator, \
    HalfAndHalfGenerator, SubtreeExchangeRecombinator, SubtreeExchangeLeafBiasedRecombinator, OnePointRecombinatorBase, \
    OnePointRecombinator, OnePointLeafBiasedRecombinator


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


class NodeCheckTestCase(TestCase):

    def _check_node(self, node, value, left_value, right_value, parent_value):
        self.assertEqual(value, node.value, 'Node values not equal')
        if not node.left:
            self.assertIsNone(left_value, 'Left value not none')
        else:
            self.assertEqual(left_value, node.left.value, 'Left values not equal')

        if not node.right:
            self.assertIsNone(right_value, 'Right value not none')
        else:
            self.assertEqual(right_value, node.right.value, 'Right values not equal')

        if not node.parent:
            self.assertIsNone(parent_value, 'Parent value not none')
        else:
            self.assertEqual(parent_value, node.parent.value, 'Parent values not equal')

    def check_root(self, parent_node, value, left_value, right_value):
        self._check_node(parent_node, value, left_value, right_value, None)

    def check_leaf(self, leaf_node, value, parent_value):
        self._check_node(leaf_node, value, None, None, parent_value)

    def check_stump(self, stump_node, value):
        self._check_node(stump_node, value, None, None, None)


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

        # test bad type
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

        # test bad type
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

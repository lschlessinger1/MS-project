from unittest import TestCase

from src.evalg.encoding import BinaryTreeNode, BinaryTree
from src.evalg.fitness import parsimony_pressure, covariant_parsimony_pressure, structural_hamming_dist


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

    def test_structural_hamming_dist_stumps(self):
        tree_1 = BinaryTree(BinaryTreeNode('*'))
        tree_2 = BinaryTree(BinaryTreeNode('*'))
        result = structural_hamming_dist(tree_1, tree_2)
        self.assertEqual(0, result)

        tree_1 = BinaryTree(BinaryTreeNode('+'))
        tree_2 = BinaryTree(BinaryTreeNode('*'))
        result = structural_hamming_dist(tree_1, tree_2)
        self.assertEqual(1, result)

    def test_structural_hamming_dist_small_trees(self):
        root_1 = BinaryTreeNode('*')
        root_1.add_left(10)
        root_1.add_right(20)
        tree_1 = BinaryTree(root_1)

        root_2 = BinaryTreeNode('+')
        root_2.add_left(10)
        root_2.add_right(30)
        tree_2 = BinaryTree(root_2)

        result = structural_hamming_dist(tree_1, tree_2)
        self.assertEqual(2 / 3, result)

    def test_structural_hamming_dist_complex_trees(self):
        #    tree 1
        #       *
        #      / \
        #    10   20
        #   /
        # 40
        root_1 = BinaryTreeNode('*')
        left = root_1.add_left(10)
        root_1.add_right(20)
        left.add_left(40)
        tree_1 = BinaryTree(root_1)

        #    tree 2
        #       +
        #      / \
        #    10   20
        #   /  \
        # 50   40
        root_2 = BinaryTreeNode('+')
        left = root_2.add_left(10)
        root_2.add_right(20)
        left.add_right(40)
        left.add_left(50)
        tree_2 = BinaryTree(root_2)

        result = structural_hamming_dist(tree_1, tree_2)
        self.assertEqual(2 / 3, result)

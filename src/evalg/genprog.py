import copy
from abc import ABC
from typing import List

import numpy as np

from src.evalg.crossover import Recombinator, check_two_parents
from src.evalg.encoding import operators, BinaryTreeNode, BinaryTree
from src.evalg.mutation import Mutator


class BinaryTreeGenerator:

    def __init__(self, binary_operators: List[str], operands, max_depth: int):
        self.binary_operators = binary_operators
        self.operands = operands
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self.max_depth = max_depth

    def generate(self):
        raise NotImplementedError('generate must be implemented in a child class.')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'binary_operators={self.binary_operators!r}, operands=' \
            f'{self.operands!r}, max_depth={self.max_depth!r})'


# Binary tree generators

class GrowGenerator(BinaryTreeGenerator):

    def __init__(self, binary_operators: List[str], operands, max_depth: int):
        super().__init__(binary_operators, operands, max_depth)

    def generate(self):
        """ Generate random binary tree

        :return:
        """
        return BinaryTree(self.grow(depth=0))

    def grow(self, depth: int):
        """Grow a random binary tree node

        :param depth: the level of the current tree
        :return:
        """
        if depth < 0:
            raise ValueError('depth must be nonnegative.')

        terminals = self.operands
        internals = self.binary_operators
        # 2 children for binary trees
        n_children = 2
        if depth < self.max_depth:
            node = BinaryTreeNode(np.random.choice(terminals + internals))

            if node.value in internals:
                for i in range(0, n_children):
                    child_i = self.grow(depth + 1)
                    if not node.left:
                        node.left = child_i
                        node.left.parent = node
                    elif not node.right:
                        node.right = child_i
                        node.right.parent = node
        else:
            node = BinaryTreeNode(np.random.choice(terminals))

        return node

    def __repr__(self):
        return f'{self.__class__.__name__}('f'binary_operators={self.binary_operators!r}, operands=' \
            f'{self.operands!r}, max_depth={self.max_depth!r})'


class FullGenerator(BinaryTreeGenerator):

    def __init__(self, binary_operators: List[str], operands, max_depth: int):
        super().__init__(binary_operators, operands, max_depth)

    def generate(self):
        return BinaryTree(self.full(depth=0))

    def full(self, depth: int):
        """ Grow a random tree"""
        if depth < 0:
            raise ValueError('depth must be nonnegative.')

        terminals = self.operands
        internals = self.binary_operators
        # 2 children for binary trees
        n_children = 2
        if depth < self.max_depth:
            node = BinaryTreeNode(np.random.choice(internals))

            if node.value in internals:
                for i in range(0, n_children):
                    child_i = self.full(depth + 1)
                    if not node.left:
                        node.left = child_i
                        node.left.parent = node
                    elif not node.right:
                        node.right = child_i
                        node.right.parent = node
        else:
            node = BinaryTreeNode(np.random.choice(terminals))

        return node

    def __repr__(self):
        return f'{self.__class__.__name__}('f'binary_operators={self.binary_operators!r}, operands=' \
            f'{self.operands!r}, max_depth={self.max_depth!r})'


class TreeMutator(Mutator, ABC):

    def __init__(self, operands):
        """

        :param operands: the possible operands to choose from
        """
        self.operands = operands

    def mutate(self, individual: BinaryTree):
        raise NotImplementedError('mutate must be implemented in a child class.')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r})'


def check_binary_trees(f):
    def wrapper(self, parents):
        if not all(isinstance(parent, BinaryTree) for parent in parents):
            raise TypeError('all parents must be of type %s' % BinaryTree.__name__)
        return f(self, parents)

    return wrapper


# Binary tree mutators

class TreePointMutator(TreeMutator):

    def __init__(self, operands):
        super().__init__(operands)

    def mutate(self, individual: BinaryTree):
        """Point mutation."""
        tree = individual

        postfix_tokens = tree.postfix_tokens()

        r = np.random.randint(0, len(postfix_tokens))
        node = tree.select_postorder(r)

        # change node value to a different value
        if node.value in self.operands:
            new_val = np.random.choice(list(set(self.operands) - {node.value}))
        elif node.value in operators:
            new_val = np.random.choice(list(set(operators) - {node.value}))
        else:
            raise TypeError('%s not in operands or operators' % node.label)

        node.value = new_val

        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r})'


class SubTreeExchangeMutator(TreeMutator, ABC):

    def __init__(self, operands, max_depth: int):
        """

        :param operands:
        :param max_depth:
        """
        super().__init__(operands)
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self.max_depth = max_depth

    @staticmethod
    def _mutate_subtree_exchange(tree: BinaryTree, tree_generator: BinaryTreeGenerator):
        """

        :param tree:
        :param tree_generator:
        :return:
        """
        postfix_tokens = tree.postfix_tokens()

        random_tree = tree_generator.generate()

        r = np.random.randint(0, len(postfix_tokens))
        new_tree = SubTreeExchangeMutator._swap_mut_subtree(tree, r, random_tree)

        return new_tree

    @staticmethod
    def _swap_mut_subtree(tree: BinaryTree, r: int, random_tree: BinaryTree):
        """ Add mutated subtree to original tree

        :param tree:
        :param r:
        :param random_tree:
        :return:
        """
        # swap parents of nodes
        node = tree.select_postorder(r)
        if node.parent:
            if node.parent.left is node:
                node.parent.left = random_tree.root
            elif node.parent.right is node:
                node.parent.right = random_tree.root
            random_tree.root.parent = node.parent
            return tree
        else:
            new_tree = BinaryTree()
            new_tree.root = random_tree.root
            return new_tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r}, max_depth={self.max_depth!r})'


class GrowMutator(SubTreeExchangeMutator):

    def __init__(self, operands, max_depth: int = 2):
        super().__init__(operands, max_depth)

    def mutate(self, individual: BinaryTree):
        tree = individual
        tree_generator = GrowGenerator(operators, self.operands, self.max_depth)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r}, max_depth={self.max_depth!r})'


class HalfAndHalfMutator(SubTreeExchangeMutator):

    def __init__(self, operands, max_depth: int = 2):
        super().__init__(operands, max_depth)

    def mutate(self, individual: BinaryTree):
        tree = individual
        tree_generator = FullGenerator(operators, self.operands, self.max_depth)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.operands!r}, max_depth={self.max_depth!r})'


# Binary tree recombinators

class SubtreeExchangeBinaryRecombinator(Recombinator):

    @check_binary_trees
    @check_two_parents
    def crossover(self, parents: List[BinaryTree]):
        """Sub-tree exchange crossover

        :return:
        """
        tree_1 = parents[0]
        tree_2 = parents[1]

        postfix_tokens_1 = tree_1.postfix_tokens()
        postfix_tokens_2 = tree_2.postfix_tokens()

        r1, r2 = SubtreeExchangeBinaryRecombinator._select_token_ind(postfix_tokens_1, postfix_tokens_2)

        # select nodes in tree
        node_1 = tree_1.select_postorder(r1)
        node_2 = tree_2.select_postorder(r2)

        SubtreeExchangeBinaryRecombinator._swap_subtrees(node_1, node_2)

        return tree_1, tree_2

    @staticmethod
    def _swap_subtrees(node_1: BinaryTreeNode, node_2: BinaryTreeNode):
        """Swap parents and of nodes
        """

        node_1_cp = copy.copy(node_1)
        node_2_cp = copy.copy(node_2)

        node_1_parent_cp = node_1_cp.parent
        node_2_parent_cp = node_2_cp.parent

        # find out if node is left or right child
        if node_1_parent_cp:
            if node_1_parent_cp.left is node_1:
                node_1.parent.left = node_2_cp
            elif node_1_parent_cp.right is node_1:
                node_1.parent.right = node_2_cp

        if node_2_parent_cp:
            if node_2_parent_cp.left is node_2:
                node_2.parent.left = node_1_cp
            elif node_2_parent_cp.right is node_2:
                node_2.parent.right = node_1_cp

        node_1.parent = node_2_parent_cp
        node_2.parent = node_1_parent_cp

        return node_1, node_2

    @staticmethod
    def _valid_pair(postfix_token_1: str, postfix_token_2: str):
        """Checks if postfix token pair is valid

        :param postfix_token_1: The first token in post-order notation
        :param postfix_token_2: The second token in post-order notation
        :return:
        """
        if postfix_token_1 in operators and postfix_token_2 in operators:
            return True
        elif postfix_token_1 not in operators and postfix_token_2 not in operators:
            return True

        return False

    @staticmethod
    def _select_token_ind(postfix_tokens_1: str, postfix_tokens_2: str):
        """Select indices of parent postfix tokens

        :param postfix_tokens_1: The first list of tokens in post-order notation
        :param postfix_tokens_2: The second list of tokens in post-order notation
        :return:
        """
        r1 = np.random.randint(0, len(postfix_tokens_1))
        r2 = np.random.randint(0, len(postfix_tokens_2))
        while not SubtreeExchangeBinaryRecombinator._valid_pair(postfix_tokens_1[r1], postfix_tokens_2[r2]):
            r1 = np.random.randint(0, len(postfix_tokens_1))
            r2 = np.random.randint(0, len(postfix_tokens_2))

        return r1, r2
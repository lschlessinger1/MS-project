from abc import ABC

import numpy as np

from src.evalg.encoding import BinaryTreeNode, BinaryTree, operators
from src.evalg.genprog.generators import BinaryTreeGenerator, GrowGenerator, FullGenerator, HalfAndHalfGenerator
from src.evalg.mutation import Mutator


class TreeMutator(Mutator, ABC):
    operands: list

    def __init__(self, operands, binary_tree_node_cls: type):
        """

        :param operands: the possible operands to choose from
        """
        self.operands = operands
        self.binary_tree_node_cls = binary_tree_node_cls

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Mutate a binary tree.

        :param individual:
        :return:
        """
        raise NotImplementedError('mutate must be implemented in a child class.')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r})'


# Binary tree mutators

class TreePointMutator(TreeMutator):
    """Node replacement mutation (also known as point mutation).

    A node in the tree is randomly selected and randomly changed, keeping the replacement node with the same number of
    arguments as the node it is replacing.
    """

    def __init__(self, operands, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(operands, binary_tree_node_cls)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Point mutation.

        :param individual:
        :return:
        """
        tree = individual

        postfix_tokens = tree.postfix_tokens()

        r = np.random.randint(0, len(postfix_tokens))
        node = tree.select_postorder(r)

        # change node value to a different value
        is_operand_type = type(node.value) in [type(op) for op in self.operands]
        if node.value in operators:
            # Node is an operator.
            new_val = np.random.choice(list(set(operators) - {node.value}))
        elif node.value in self.operands or is_operand_type:
            # Node is an operand.
            new_val = np.random.choice(list(set(self.operands) - {node.value}))
        else:
            raise TypeError('%s not in operands or operators' % node.label)

        node.value = new_val

        return tree


class SubTreeExchangeMutator(TreeMutator, ABC):
    max_depth: int

    def __init__(self, operands, max_depth, binary_tree_node_cls: type):
        """

        :param operands:
        :param max_depth:
        """
        super().__init__(operands, binary_tree_node_cls)
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self.max_depth = max_depth

    @staticmethod
    def _mutate_subtree_exchange(tree: BinaryTree,
                                 tree_generator: BinaryTreeGenerator) -> BinaryTree:
        """Mutate sub-tree exchange.

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
    def _swap_mut_subtree(tree: BinaryTree,
                          r: int,
                          random_tree: BinaryTree) -> BinaryTree:
        """Add mutated subtree to original tree.

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

    def __init__(self, operands, max_depth=2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(operands, max_depth, binary_tree_node_cls)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Mutate a grown binary tree.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = GrowGenerator(operators, self.operands, self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree


class FullMutator(SubTreeExchangeMutator):

    def __init__(self, operands, max_depth=2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(operands, max_depth, binary_tree_node_cls)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Full mutation applied to a binary tree

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = FullGenerator(operators, self.operands, self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree


class HalfAndHalfMutator(SubTreeExchangeMutator):
    """Ramped half-and-half method

    Koza, John R., and John R. Koza. Genetic programming: on the programming of computers by means of natural
    selection. Vol. 1. MIT press, 1992.
    """

    def __init__(self, operands, max_depth=2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(operands, max_depth, binary_tree_node_cls)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Half and half mutation applied to a binary tree.

        Half of the time, the tree is generated using the grow method, and the other half of the time is generated
        using the full method.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = HalfAndHalfGenerator(operators, self.operands, self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

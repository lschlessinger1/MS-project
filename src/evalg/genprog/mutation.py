import importlib
from abc import ABC

import numpy as np

from src.evalg.encoding import BinaryTreeNode, BinaryTree
from src.evalg.genprog.generators import BinaryTreeGenerator, GrowGenerator, FullGenerator, HalfAndHalfGenerator
from src.evalg.serialization import Serializable


class TreeMutator(Serializable):

    def __init__(self, binary_tree_node_cls: type):
        """

        """
        self.binary_tree_node_cls = binary_tree_node_cls

    def mutate(self,
               operators: list,
               operands: list,
               individual: BinaryTree) -> BinaryTree:
        """Mutate a binary tree.

        :param individual:
        :return:
        """
        raise NotImplementedError('mutate must be implemented in a child class.')

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["binary_tree_node_module_name"] = self.binary_tree_node_cls.__module__
        input_dict["binary_tree_node_cls_name"] = self.binary_tree_node_cls.__qualname__
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)

        b_tree_module_name = input_dict.pop("binary_tree_node_module_name")
        b_tree_cls_name = input_dict.pop("binary_tree_node_cls_name")
        b_tree_module = importlib.import_module(b_tree_module_name)
        binary_tree_node_cls = getattr(b_tree_module, b_tree_cls_name)
        input_dict["binary_tree_node_cls"] = binary_tree_node_cls
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}'


# Binary tree mutators

class TreePointMutator(TreeMutator):
    """Node replacement mutation (also known as point mutation).

    A node in the tree is randomly selected and randomly changed, keeping the replacement node with the same number of
    arguments as the node it is replacing.
    """

    def __init__(self, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(binary_tree_node_cls)

    def mutate(self,
               operators: list,
               operands: list,
               individual: BinaryTree) -> BinaryTree:
        """Point mutation.

        :param individual:
        :return:
        """
        tree = individual

        postfix_tokens = tree.postfix_tokens()

        r = np.random.randint(0, len(postfix_tokens))
        node = tree.select_postorder(r)

        # change node value to a different value
        is_operand_type = type(node.value) in [type(op) for op in operands]
        if node.value in operators:
            # Node is an operator.
            new_val = np.random.choice(list(set(operators) - {node.value}))
        elif node.value in operands or is_operand_type:
            # Node is an operand.
            new_val = np.random.choice(list(set(operands) - {node.value}))
        else:
            raise TypeError('%s not in operands or operators' % node.label)

        node.value = new_val

        return tree


class SubTreeExchangeMutator(TreeMutator, ABC):

    def __init__(self, max_depth: int, binary_tree_node_cls: type):
        """

        :param max_depth:
        """
        super().__init__(binary_tree_node_cls)
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self.max_depth = max_depth

    @staticmethod
    def _mutate_subtree_exchange(
            operators: list,
            operands: list,
            tree: BinaryTree,
            tree_generator: BinaryTreeGenerator) -> BinaryTree:
        """Mutate sub-tree exchange.

        :param tree:
        :param tree_generator:
        :return:
        """
        postfix_tokens = tree.postfix_tokens()

        random_tree = tree_generator.generate(operators, operands)

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

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["max_depth"] = self.max_depth
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'max_depth={self.max_depth!r})'


class GrowMutator(SubTreeExchangeMutator):

    def __init__(self, max_depth: int = 2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def mutate(self,
               operators: list,
               operands: list,
               individual: BinaryTree) -> BinaryTree:
        """Mutate a grown binary tree.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = GrowGenerator(self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(operators, operands, tree, tree_generator)
        return tree


class FullMutator(SubTreeExchangeMutator):

    def __init__(self, max_depth: int = 2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def mutate(self,
               operators: list,
               operands: list,
               individual: BinaryTree) -> BinaryTree:
        """Full mutation applied to a binary tree

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = FullGenerator(self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(operators, operands, tree, tree_generator)
        return tree


class HalfAndHalfMutator(SubTreeExchangeMutator):
    """Ramped half-and-half method

    Koza, John R., and John R. Koza. Genetic programming: on the programming of computers by means of natural
    selection. Vol. 1. MIT press, 1992.
    """

    def __init__(self, max_depth: int = 2, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def mutate(self,
               operators: list,
               operands: list,
               individual: BinaryTree) -> BinaryTree:
        """Half and half mutation applied to a binary tree.

        Half of the time, the tree is generated using the grow method, and the other half of the time is generated
        using the full method.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = HalfAndHalfGenerator(self.max_depth, self.binary_tree_node_cls)
        tree = self._mutate_subtree_exchange(operators, operands, tree, tree_generator)
        return tree

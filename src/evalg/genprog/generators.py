import importlib
from typing import List

import numpy as np

from src.evalg.encoding import BinaryTree, BinaryTreeNode
from src.evalg.serialization import Serializable


class BinaryTreeGenerator(Serializable):
    _max_depth: int

    def __init__(self, max_depth: int, binary_tree_node_cls: type):
        self._max_depth = max_depth
        self.binary_tree_node_cls = binary_tree_node_cls

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth: int) -> None:
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self._max_depth = max_depth

    def generate(self, binary_operators: List[str], operands: list) -> BinaryTree:
        """Generate a binary tree.

        :return:
        """
        raise NotImplementedError('generate must be implemented in a child class.')

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["max_depth"] = self.max_depth
        input_dict["binary_tree_node_module_name"] = self.binary_tree_node_cls.__module__
        input_dict["binary_tree_node_cls_name"] = self.binary_tree_node_cls.__qualname__
        return input_dict

    @staticmethod
    def from_dict(input_dict: dict):
        b_tree_module_name = input_dict.pop("binary_tree_node_module_name")
        b_tree_cls_name = input_dict.pop("binary_tree_node_cls_name")
        b_tree_module = importlib.import_module(b_tree_module_name)
        binary_tree_node_cls = getattr(b_tree_module, b_tree_cls_name)
        input_dict["binary_tree_node_cls"] = binary_tree_node_cls

        return Serializable.from_dict(input_dict)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'max_depth={self.max_depth!r})'


# Binary tree generators

class GrowGenerator(BinaryTreeGenerator):

    def __init__(self, max_depth: int, binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def generate(self,
                 binary_operators: List[str],
                 operands: list) -> BinaryTree:
        """Grow a random binary tree.

        :return:
        """
        return BinaryTree(self.grow(binary_operators, operands, depth=0))

    def grow(self,
             binary_operators: List[str],
             operands: list,
             depth: int) -> BinaryTreeNode:
        """Grow a random binary tree node.

        Generate trees of different sizes and shapes. Nodes are chosen from the primitive set until the maximum tree
        depth is reached. Greater than that depth, terminals are selected.

        :param depth: the level of the current tree
        :return:
        """
        if depth < 0:
            raise ValueError('depth must be nonnegative.')

        terminals = operands
        internals = binary_operators
        # 2 children for binary trees
        n_children = 2
        if depth < self.max_depth:
            primitives = terminals + internals
            node = self.binary_tree_node_cls(np.random.choice(primitives))

            if node.value in internals:
                for i in range(0, n_children):
                    child_i = self.grow(binary_operators, operands, depth + 1)
                    if not node.left:
                        node.left = child_i
                        node.left.parent = node
                    elif not node.right:
                        node.right = child_i
                        node.right.parent = node
        else:
            node = self.binary_tree_node_cls(np.random.choice(terminals))

        return node


class FullGenerator(BinaryTreeGenerator):

    def __init__(self,
                 max_depth: int,
                 binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def generate(self,
                 binary_operators: List[str],
                 operands: list) -> BinaryTree:
        """Generate a full binary tree.

        :return:
        """
        return BinaryTree(self.full(binary_operators, operands, depth=0))

    def full(self,
             binary_operators: List[str],
             operands: list,
             depth: int) -> BinaryTreeNode:
        """Grow a random tree.

         Generates full trees, i.e. all leaves having the same depth. Nodes are chosen uniformly at random from the
         internals until the maximum tree depth is reached. Greater than that depth, terminals are selected.

        :param depth:
        :return:
        """
        if depth < 0:
            raise ValueError('depth must be nonnegative.')

        terminals = operands
        internals = binary_operators
        # 2 children for binary trees
        n_children = 2
        if depth < self.max_depth:
            node = self.binary_tree_node_cls(np.random.choice(internals))

            if node.value in internals:
                for i in range(0, n_children):
                    child_i = self.full(binary_operators, operands, depth + 1)
                    if not node.left:
                        node.left = child_i
                        node.left.parent = node
                    elif not node.right:
                        node.right = child_i
                        node.right.parent = node
        else:
            node = self.binary_tree_node_cls(np.random.choice(terminals))

        return node


class HalfAndHalfGenerator(BinaryTreeGenerator):

    def __init__(self,
                 max_depth: int,
                 binary_tree_node_cls: type = BinaryTreeNode):
        super().__init__(max_depth, binary_tree_node_cls)

    def generate(self,
                 binary_operators: List[str],
                 operands: list) -> BinaryTree:
        """Generate a full binary tree.

        :return:
        """
        if np.random.rand() > 0.5:
            tree_generator = FullGenerator(self.max_depth, self.binary_tree_node_cls)
            root = tree_generator.full(binary_operators, operands, depth=0)
        else:
            tree_generator = GrowGenerator(self.max_depth, self.binary_tree_node_cls)
            root = tree_generator.grow(binary_operators, operands, depth=0)
        return BinaryTree(root)

from abc import ABC
from typing import List, Callable, Tuple, Optional

import numpy as np

from src.evalg.crossover import Recombinator, check_two_parents
from src.evalg.encoding import operators, BinaryTreeNode, BinaryTree
from src.evalg.mutation import Mutator


class BinaryTreeGenerator:
    binary_operators: List[str]
    operands: list
    _max_depth: int

    def __init__(self, binary_operators, operands, max_depth: int):
        self.binary_operators = binary_operators
        self.operands = operands
        self._max_depth = max_depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth: int) -> None:
        if max_depth < 0:
            raise ValueError('max depth must be nonnegative')
        self._max_depth = max_depth

    def generate(self) -> BinaryTree:
        """Generate a binary tree.

        :return:
        """
        raise NotImplementedError('generate must be implemented in a child class.')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'binary_operators={self.binary_operators!r}, operands=' \
            f'{self.operands!r}, max_depth={self.max_depth!r})'


# Binary tree generators

class GrowGenerator(BinaryTreeGenerator):

    def __init__(self, binary_operators, operands, max_depth: int):
        super().__init__(binary_operators, operands, max_depth)

    def generate(self) -> BinaryTree:
        """Grow a random binary tree.

        :return:
        """
        return BinaryTree(self.grow(depth=0))

    def grow(self, depth: int) -> BinaryTreeNode:
        """Grow a random binary tree node.

        Generate trees of different sizes and shapes. Nodes are chosen from the primitive set until the maximum tree
        depth is reached. Greater than that depth, terminals are selected.

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
            primitives = terminals + internals
            node = BinaryTreeNode(np.random.choice(primitives))

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

    def __init__(self, binary_operators, operands, max_depth: int):
        super().__init__(binary_operators, operands, max_depth)

    def generate(self) -> BinaryTree:
        """Generate a full binary tree.

        :return:
        """
        return BinaryTree(self.full(depth=0))

    def full(self, depth: int) -> BinaryTreeNode:
        """Grow a random tree.

         Generates full trees, i.e. all leaves having the same depth. Nodes are chosen uniformly at random from the
         internals until the maximum tree depth is reached. Greater than that depth, terminals are selected.

        :param depth:
        :return:
        """
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


class HalfAndHalfGenerator(BinaryTreeGenerator):

    def __init__(self, binary_operators, operands, max_depth: int):
        super().__init__(binary_operators, operands, max_depth)

    def generate(self) -> BinaryTree:
        """Generate a full binary tree.

        :return:
        """
        if np.random.rand() > 0.5:
            tree_generator = FullGenerator(operators, self.operands, self.max_depth)
            root = tree_generator.full(depth=0)
        else:
            tree_generator = GrowGenerator(operators, self.operands, self.max_depth)
            root = tree_generator.grow(depth=0)
        return BinaryTree(root)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'binary_operators={self.binary_operators!r}, operands=' \
            f'{self.operands!r}, max_depth={self.max_depth!r})'


class TreeMutator(Mutator, ABC):
    operands: list

    def __init__(self, operands):
        """

        :param operands: the possible operands to choose from
        """
        self.operands = operands

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Mutate a binary tree.

        :param individual:
        :return:
        """
        raise NotImplementedError('mutate must be implemented in a child class.')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r})'


def check_binary_trees(f: Callable) -> Callable:
    def wrapper(self, parents: List[BinaryTree]):
        if not all(isinstance(parent, BinaryTree) for parent in parents):
            raise TypeError('all parents must be of type %s' % BinaryTree.__name__)
        return f(self, parents)

    return wrapper


# Binary tree mutators

class TreePointMutator(TreeMutator):
    """Node replacement mutation (also known as point mutation).

    A node in the tree is randomly selected and randomly changed, keeping the replacement node with the same number of
    arguments as the node it is replacing.
    """

    def __init__(self, operands):
        super().__init__(operands)

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

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r})'


class SubTreeExchangeMutator(TreeMutator, ABC):
    max_depth: int

    def __init__(self, operands, max_depth):
        """

        :param operands:
        :param max_depth:
        """
        super().__init__(operands)
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

    def __init__(self, operands, max_depth=2):
        super().__init__(operands, max_depth)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Mutate a grown binary tree.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = GrowGenerator(operators, self.operands, self.max_depth)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operands={self.operands!r}, max_depth={self.max_depth!r})'


class FullMutator(SubTreeExchangeMutator):

    def __init__(self, operands, max_depth=2):
        super().__init__(operands, max_depth)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Full mutation applied to a binary tree

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = FullGenerator(operators, self.operands, self.max_depth)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.operands!r}, max_depth={self.max_depth!r})'


class HalfAndHalfMutator(SubTreeExchangeMutator):
    """Ramped half-and-half method

    Koza, John R., and John R. Koza. Genetic programming: on the programming of computers by means of natural
    selection. Vol. 1. MIT press, 1992.
    """

    def __init__(self, operands, max_depth=2):
        super().__init__(operands, max_depth)

    def mutate(self, individual: BinaryTree) -> BinaryTree:
        """Half and half mutation applied to a binary tree.

        Half of the time, the tree is generated using the grow method, and the other half of the time is generated
        using the full method.

        :param individual:
        :return:
        """
        tree = individual
        tree_generator = HalfAndHalfGenerator(operators, self.operands, self.max_depth)
        tree = self._mutate_subtree_exchange(tree, tree_generator)
        return tree

    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.operands!r}, max_depth={self.max_depth!r})'


# Binary tree recombinators

class SubtreeExchangeRecombinatorBase(Recombinator, ABC):

    @staticmethod
    def _swap_subtrees(node_1: BinaryTreeNode,
                       node_2: BinaryTreeNode,
                       tree_1: BinaryTree,
                       tree_2: BinaryTree) -> None:
        """Swap parents and children of nodes.


        :param node_1:
        :param node_2:
        :param tree_1: tree corresponding to node 1
        :param tree_2: tree corresponding to node 2
        :return:
        """
        if node_1 == node_2:
            return

        if node_1 is None or node_2 is None:
            return

        if not node_1.has_parent() and not node_2.has_parent():
            return

        if not node_1.has_parent():
            tree_1.root = node_2
            if node_2.is_left_child():
                node_2.parent.left = node_1
            else:
                node_2.parent.right = node_1
        elif not node_2.has_parent():
            tree_2.root = node_1
            if node_1.is_left_child():
                node_1.parent.left = node_2
            else:
                node_1.parent.right = node_2
        else:
            if node_1.is_left_child():
                if node_2.is_left_child():
                    node_2.parent.left, node_1.parent.left = node_1, node_2
                else:
                    node_2.parent.right, node_1.parent.left = node_1, node_2
            else:
                if node_2.is_left_child():
                    node_2.parent.left, node_1.parent.right = node_1, node_2
                else:
                    node_2.parent.right, node_1.parent.right = node_1, node_2

        node_1.parent, node_2.parent = node_2.parent, node_1.parent

    @staticmethod
    def _valid_pair(token_1: str,
                    token_2: str) -> bool:
        """Checks if token pair is valid.

        :param token_1: The first token
        :param token_2: The second token
        :return:
        """
        if token_1 in operators and token_2 in operators:
            return True
        elif token_1 not in operators and token_2 not in operators:
            return True

        return False


class SubtreeExchangeRecombinator(SubtreeExchangeRecombinatorBase):

    @staticmethod
    def _select_token_ind(tokens_1: List[str],
                          tokens_2: List[str]) -> Tuple[int, int]:
        """Select indices of parent tokens.

        :param tokens_1: The first list of tokens
        :param tokens_2: The second list of tokens
        :return:
        """
        r1 = np.random.randint(0, len(tokens_1))
        r2 = np.random.randint(0, len(tokens_2))
        while not SubtreeExchangeRecombinatorBase._valid_pair(tokens_1[r1], tokens_2[r2]):
            r1 = np.random.randint(0, len(tokens_1))
            r2 = np.random.randint(0, len(tokens_2))

        return r1, r2

    @check_binary_trees
    @check_two_parents
    def crossover(self, parents: List[BinaryTree]) -> Tuple[BinaryTree, BinaryTree]:
        """Subtree exchange crossover.

        Nodes are selected uniformly at random.

        :return:
        """
        tree_1 = parents[0]
        tree_2 = parents[1]

        postfix_tokens_1 = tree_1.postfix_tokens()
        postfix_tokens_2 = tree_2.postfix_tokens()

        r1, r2 = self._select_token_ind(postfix_tokens_1, postfix_tokens_2)

        # select nodes in tree
        node_1 = tree_1.select_postorder(r1)
        node_2 = tree_2.select_postorder(r2)

        self._swap_subtrees(node_1, node_2, tree_1, tree_2)

        return tree_1, tree_2


class SubtreeExchangeLeafBiasedRecombinator(SubtreeExchangeRecombinatorBase):
    t_prob: float

    def __init__(self, t_prob: float = 0.1):
        self.t_prob = t_prob  # probability of choosing a terminal node (leaf).

    @check_binary_trees
    @check_two_parents
    def crossover(self, parents: List[BinaryTree]) -> Tuple[BinaryTree, BinaryTree]:
        """Subtree exchange leaf biased crossover.

        Select terminal nodes with probability t_prob and internal nodes with probability 1 - t_prob.

        :return:
        """
        tree_1 = parents[0]
        tree_2 = parents[1]

        postfix_tokens_1 = tree_1.postfix_tokens()
        postfix_tokens_2 = tree_2.postfix_tokens()

        tree_1_single_elt = len(postfix_tokens_1) == 1
        tree_2_single_elt = len(postfix_tokens_2) == 1
        if tree_1_single_elt and tree_2_single_elt:
            return tree_1, tree_2

        # If either tree has a single node, we must select a terminal node.
        either_tree_single_elt = tree_1_single_elt or tree_2_single_elt
        if self.t_prob > np.random.rand() or either_tree_single_elt:
            # Choose terminal node pair uniformly at random.
            terminals_1 = [i for (i, token) in enumerate(postfix_tokens_1) if token not in operators]
            terminals_2 = [i for (i, token) in enumerate(postfix_tokens_2) if token not in operators]
            r1 = np.random.choice(terminals_1)
            r2 = np.random.choice(terminals_2)
        else:
            # Choose internal node pair uniformly at random.
            internals_1 = [i for (i, token) in enumerate(postfix_tokens_1) if token in operators]
            internals_2 = [i for (i, token) in enumerate(postfix_tokens_2) if token in operators]
            r1 = np.random.choice(internals_1)
            r2 = np.random.choice(internals_2)

        # Select nodes in tree.
        node_1 = tree_1.select_postorder(r1)
        node_2 = tree_2.select_postorder(r2)

        self._swap_subtrees(node_1, node_2, tree_1, tree_2)

        return tree_1, tree_2


NodePair = Tuple[BinaryTreeNode, BinaryTreeNode]


class OnePointRecombinator(SubtreeExchangeRecombinatorBase):

    @check_binary_trees
    @check_two_parents
    def crossover(self, parents: list) -> Tuple[BinaryTree, BinaryTree]:
        """One point crossover.

        :param parents:
        :return:
        """
        tree_1 = parents[0]
        tree_2 = parents[1]

        # All the nodes encountered are stored, forming a tree fragment
        common_region = self.get_common_region(tree_1.root, tree_2.root)

        if len(common_region) <= 1:
            return tree_1, tree_2

        node_1, node_2 = self.select_node_pair(common_region)

        self._swap_subtrees(node_1, node_2, tree_1, tree_2)

        return tree_1, tree_2

    def _get_common_region(self,
                           node_1: BinaryTreeNode,
                           node_2: BinaryTreeNode,
                           valid_pairs: Optional[List[NodePair]] = None) -> None:
        """Recursive helper to get common region."""
        if valid_pairs is None:
            valid_pairs = []

        if node_1 and node_2:
            both_leaves = node_1.is_leaf() and node_2.is_leaf()
            both_internals = not node_1.is_leaf() and not node_2.is_leaf()
            if both_leaves or both_internals:
                valid_pairs.append((node_1, node_2))
                self._get_common_region(node_1.left, node_2.left, valid_pairs)
                self._get_common_region(node_1.right, node_2.right, valid_pairs)

    def get_common_region(self,
                          node_1: BinaryTreeNode,
                          node_2: BinaryTreeNode) -> List[NodePair]:
        """Get valid pairs of nodes using an in-order traversal"""
        common_region = []
        self._get_common_region(node_1, node_2, common_region)
        return common_region

    @staticmethod
    def select_node_pair(common_region: List[NodePair]) -> NodePair:
        """A random crossover point is selected with a uniform probability.

        :param common_region:
        :return: A pair of nodes representing a random crossover point
        """
        r = np.random.randint(0, len(common_region))
        return common_region[r]


class OnePointLeafBiasedRecombinator(SubtreeExchangeRecombinatorBase):
    t_prob: float

    def __init__(self, t_prob: float = 0.1):
        self.t_prob = t_prob  # probability of choosing a terminal node (leaf).

    @check_binary_trees
    @check_two_parents
    def crossover(self, parents: list) -> Tuple[BinaryTree, BinaryTree]:
        pass

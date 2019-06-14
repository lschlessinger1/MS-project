from abc import ABC
from typing import List, Tuple, Optional

import numpy as np

from src.evalg.crossover import Recombinator, check_two_parents
from src.evalg.encoding import BinaryTreeNode, BinaryTree, operators
from src.evalg.genprog.util import check_binary_trees


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
        choose_terminal = self.t_prob < np.random.rand()
        if choose_terminal or either_tree_single_elt:
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


class OnePointRecombinatorBase(SubtreeExchangeRecombinatorBase):
    """Base class for one point crossover"""

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

        selected_nodes = self.select_node_pair(common_region)
        if selected_nodes is None:
            return tree_1, tree_2
        else:
            node_1, node_2 = selected_nodes

        self._swap_subtrees(node_1, node_2, tree_1, tree_2)

        return tree_1, tree_2

    def select_node_pair(self, common_region: List[NodePair]) -> NodePair:
        """A crossover point is selected.

        :param common_region:
        :return: A pair of nodes representing a random crossover point
        """
        raise NotImplementedError('Select node pair must be implemented in a child class.')


class OnePointRecombinator(OnePointRecombinatorBase):

    def select_node_pair(self, common_region: List[NodePair]) -> NodePair:
        """A random crossover point is selected with a uniform probability."""
        if len(common_region) > 0:
            r = np.random.randint(0, len(common_region))
            return common_region[r]


class OnePointStrictRecombinator(OnePointRecombinatorBase):
    """Strict one-point crossover.

    Poli and Langdon, 1997b
    """

    @staticmethod
    def get_allowed_pairs(common_region: List[NodePair]) -> List[NodePair]:
        terminals = [(node_1, node_2) for (i, (node_1, node_2)) in enumerate(common_region)
                     if node_1.value not in operators and node_2.value not in operators]
        internals = [(node_1, node_2) for (i, (node_1, node_2)) in enumerate(common_region)
                     if node_1.value in operators and node_2.value in operators]

        # Pairs the have the same internal/operator
        same_functions = [(node_1, node_2) for (node_1, node_2) in internals if node_1.value == node_2.value]

        return same_functions + terminals

    def select_node_pair(self, common_region: List[NodePair]) -> NodePair:
        """Exactly like one-point crossover except that the
        crossover point can be located only in the parts of the two
        trees which are exactly the same (i.e. which have the same
        functions in the nodes encountered traversing the trees from
        the root node).
        """
        allowed_node_pairs = self.get_allowed_pairs(common_region)
        recombinator = OnePointRecombinator()
        return recombinator.select_node_pair(allowed_node_pairs)


class OnePointLeafBiasedRecombinator(OnePointRecombinatorBase):
    t_prob: float

    def __init__(self, t_prob: float = 0.1):
        self.t_prob = t_prob  # probability of choosing a terminal node (leaf).

    def select_node_pair(self, common_region: List[NodePair]) -> NodePair:
        if len(common_region) == 1:
            return common_region[0]

        terminals = [i for (i, (node_1, node_2)) in enumerate(common_region)
                     if node_1.value not in operators and node_2.value not in operators]
        internals = [i for (i, (node_1, node_2)) in enumerate(common_region)
                     if node_1.value in operators and node_2.value in operators]

        if len(terminals) == 0:
            # Choose internal node pair uniformly at random.
            r = np.random.choice(internals)
            return common_region[r]
        elif len(internals) == 0:
            # Choose terminal node pair uniformly at random.
            r = np.random.choice(terminals)
            return common_region[r]
        elif len(terminals) > 0 and len(internals) > 0:
            # Choose terminal (leaf) with probability `t_prob`.
            choose_terminal = self.t_prob < np.random.rand()
            if choose_terminal:
                # Choose terminal node pair uniformly at random.
                r = np.random.choice(terminals)
                return common_region[r]
            else:
                # Choose internal node pair uniformly at random.
                r = np.random.choice(internals)
                return common_region[r]


class OnePointStrictLeafBiasedRecombinator(OnePointRecombinatorBase):
    t_prob: float

    def __init__(self, t_prob: float = 0.1):
        self.t_prob = t_prob  # probability of choosing a terminal node (leaf).

    def select_node_pair(self, common_region: List[NodePair]) -> NodePair:
        allowed_node_pairs = OnePointStrictRecombinator.get_allowed_pairs(common_region)
        recombinator = OnePointLeafBiasedRecombinator(t_prob=self.t_prob)
        return recombinator.select_node_pair(allowed_node_pairs)

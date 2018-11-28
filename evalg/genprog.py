import copy

import numpy as np

from evalg.encoding import operators, infix_to_postfix, postfix_to_binexp_tree


def crossover_subtree_exchange(infix_1, infix_2):
    postfix_1 = infix_to_postfix(infix_1)
    postfix_2 = infix_to_postfix(infix_2)

    tree_1 = postfix_to_binexp_tree(postfix_1)
    tree_2 = postfix_to_binexp_tree(postfix_2)

    postfix_tokens_1 = postfix_1.split(' ')
    postfix_tokens_2 = postfix_2.split(' ')

    r1, r2 = _select_token_ind(postfix_tokens_1, postfix_tokens_2)

    # select nodes in tree
    node_1 = tree_1.select_postorder(r1)
    node_2 = tree_2.select_postorder(r2)

    _swap_subtrees(node_1, node_2)

    return tree_1.infix(), tree_2.infix()


def _swap_subtrees(node_1, node_2):
    """Swap parents and of nodes
    """
    #
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


def _valid_pair(postfix_tokens_a, postfix_tokens_b, r1, r2):
    """ Checks if postfix token pair is valid
    """
    if postfix_tokens_a[r1] in operators and postfix_tokens_b[r2] in operators:
        return True
    elif postfix_tokens_a[r1] not in operators and postfix_tokens_b[r2] not in operators:
        return True

    return False


def _select_token_ind(postfix_tokens_1, postfix_tokens_2):
    """Select indices of parent postfix tokens
    """

    r1 = np.random.randint(0, len(postfix_tokens_1))
    r2 = np.random.randint(0, len(postfix_tokens_2))
    while not _valid_pair(postfix_tokens_1, postfix_tokens_2, r1, r2):
        r1 = np.random.randint(0, len(postfix_tokens_1))
        r2 = np.random.randint(0, len(postfix_tokens_2))

    return r1, r2

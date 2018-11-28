import copy

import numpy as np

from evalg.encoding import operators, infix_to_postfix, postfix_to_binexp_tree, BinaryTreeNode

# todo : get rid of this
kernels = ['SE1', 'SE2', 'SE3', 'SE4', 'RQ1', 'RQ2', 'RQ3', 'RQ4']


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


def grow(depth, max_depth=2):
    """ Grow a random tree
    """
    terminals = kernels
    internals = operators
    # 2 children for binary trees
    n_children = 2
    if depth < max_depth:
        node = BinaryTreeNode(np.random.choice(terminals + internals))

        if node.value in internals:
            for i in range(0, n_children):
                child_i = grow(depth + 1)
                if not node.left:
                    node.left = child_i
                    node.left.parent = node
                elif not node.right:
                    node.right = child_i
                    node.right.parent = node
    else:
        node = BinaryTreeNode(np.random.choice(terminals))

    return node


def generate_grow(max_depth=2):
    return grow(depth=0, max_depth=max_depth)


def full(depth, max_depth=2):
    """ Grow a random tree
    """
    terminals = kernels
    internals = operators
    # 2 children for binary trees
    n_children = 2
    if depth < max_depth:
        node = BinaryTreeNode(np.random.choice(internals))

        if node.value in internals:
            for i in range(0, n_children):
                child_i = full(depth + 1)
                if not node.left:
                    node.left = child_i
                    node.left.parent = node
                elif not node.right:
                    node.right = child_i
                    node.right.parent = node
    else:
        node = BinaryTreeNode(np.random.choice(terminals))

    return node


def generate_full(max_depth=2):
    return full(depth=0, max_depth=max_depth)


def _swap_mut_subtree(node, random_tree):
    # add mutated subtree to original
    # swap parents of nodes
    if node.parent:
        if node.parent.left is node:
            node.parent.left = random_tree
        elif node.parent.right is node:
            node.parent.right = random_tree
    random_tree.parent = node.parent

    return node, random_tree


def _mutate_subtree_exchange(infix, init_method, **init_method_kwargs):
    """
    init_method: initialization method
    """
    postfix = infix_to_postfix(infix)
    tree = postfix_to_binexp_tree(postfix)
    postfix_tokens = postfix.split(' ')

    r = np.random.randint(0, len(postfix_tokens))
    node = tree.select_postorder(r)

    random_tree = init_method(init_method_kwargs)

    _swap_mut_subtree(node, random_tree)

    return tree.infix()


def mutate_grow(infix, max_depth=2):
    return _mutate_subtree_exchange(infix, generate_grow, max_depth=max_depth)


def mutate_half_and_half(infix, max_depth=2):
    return _mutate_subtree_exchange(infix, generate_full, max_depth=max_depth)


def mutate_point(infix):
    """Point mutation

    :param infix:
    :return:
    """
    postfix = infix_to_postfix(infix)
    tree = postfix_to_binexp_tree(postfix)
    postfix_tokens = postfix.split(' ')

    r = np.random.randint(0, len(postfix_tokens))
    node = tree.select_postorder(r)

    # change node value to a different value
    if node.value in kernels:
        new_val = np.random.choice(list(set(kernels) - {node.value}))
    elif node.value in operators:
        new_val = np.random.choice(list(set(operators) - {node.value}))
    else:
        raise ValueError('%s not in kernels or operators' % node.value)

    node.value = new_val

    return tree.infix()

from typing import Union, Optional

from src.autoks.backend.kernel import subkernel_expression, RawKernelType, kernel_to_infix_tokens
from src.evalg.encoding import BinaryTreeNode, BinaryTree, postfix_tokens_to_binexp_tree, infix_tokens_to_postfix_tokens


class KernelNode(BinaryTreeNode):
    _value: Union[RawKernelType, str]

    def __init__(self,
                 value: Union[RawKernelType, str],
                 parent: Optional[BinaryTreeNode] = None,
                 left: Optional[BinaryTreeNode] = None,
                 right: Optional[BinaryTreeNode] = None):
        super().__init__(value, parent, left, right)

    def _value_to_label(self, value: Union[RawKernelType, str]) -> str:
        if isinstance(value, RawKernelType):
            return subkernel_expression(value)
        else:
            return str(value)

    def _value_to_html(self, value) -> str:
        if isinstance(value, RawKernelType):
            return subkernel_expression(value, html_like=True)
        else:
            return '<' + str(value) + '>'


class KernelTree(BinaryTree):
    _root: Optional[KernelNode]

    def __init__(self, root: Optional[KernelNode] = None):
        super().__init__(root)


def kernel_to_tree(kernel: RawKernelType) -> KernelTree:
    infix_tokens = kernel_to_infix_tokens(kernel)
    postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
    return postfix_tokens_to_binexp_tree(postfix_tokens, bin_tree_node_cls=KernelNode, bin_tree_cls=KernelTree)


def apply_op(left: RawKernelType,
             right: RawKernelType,
             operator: str) -> RawKernelType:
    """Apply binary operator to two gp_models.

    :param left:
    :param right:
    :param operator:
    :return:
    """
    if operator == '+':
        return left + right
    elif operator == '*':
        return left * right
    else:
        raise ValueError(f'Unknown operator {operator}')


def eval_binexp_tree(root: BinaryTreeNode) -> RawKernelType:
    """Evaluate a binary expression tree.

    :param root:
    :return:
    """
    if root is not None:
        if isinstance(root.value, RawKernelType):
            return root.value

        left_node = eval_binexp_tree(root.left)
        right_node = eval_binexp_tree(root.right)

        operator = root.value

        return apply_op(left_node, right_node, operator)


def tree_to_kernel(tree: BinaryTree) -> RawKernelType:
    """Convert a binary tree to a kernel.

    :param tree:
    :return:
    """
    return eval_binexp_tree(tree.root)


def hd_kern_nodes(node_1: KernelNode,
                  node_2: KernelNode) -> float:
    """Hamming distance between two kernel nodes

    0 if node_1 = node_2 (Both terminal nodes of equal active dim. and class)
    1 otherwise (different terminal node type or internal node)

    :param node_1: The first kernel node.
    :param node_2: The second kernel node.
    :return: The Hamming distance between nodes.
    """
    if node_1.is_leaf() and node_2.is_leaf():
        kern_1 = node_1.value
        kern_2 = node_2.value
        same_dims = kern_1.active_dims == kern_2.active_dims
        same_cls = kern_1.__class__ == kern_2.__class__
        # consider gp_models equal if they have the same active dimension and class
        nodes_eq = same_dims and same_cls
        if nodes_eq:
            return 0
        else:
            return 1
    else:
        return 1

from typing import Union, Optional

from GPy.kern import Kern

from src.autoks.backend.kernel import kernel_to_infix_tokens
from src.autoks.core.kernel import subkernel_expression
from src.evalg.encoding import BinaryTreeNode, BinaryTree, infix_tokens_to_postfix_tokens, postfix_tokens_to_binexp_tree


class KernelNode(BinaryTreeNode):
    _value: Union[Kern, str]

    def __init__(self,
                 value: Union[Kern, str],
                 parent: Optional[BinaryTreeNode] = None,
                 left: Optional[BinaryTreeNode] = None,
                 right: Optional[BinaryTreeNode] = None):
        super().__init__(value, parent, left, right)

    def _value_to_label(self, value: Union[Kern, str]) -> str:
        if isinstance(value, Kern):
            return subkernel_expression(value)
        else:
            return str(value)

    def _value_to_html(self, value) -> str:
        if isinstance(value, Kern):
            return subkernel_expression(value, html_like=True)
        else:
            return '<' + str(value) + '>'


class KernelTree(BinaryTree):
    _root: Optional[KernelNode]

    def __init__(self, root: Optional[KernelNode] = None):
        super().__init__(root)


def kernel_to_tree(kernel: Kern) -> KernelTree:
    infix_tokens = kernel_to_infix_tokens(kernel)
    postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
    return postfix_tokens_to_binexp_tree(postfix_tokens, bin_tree_node_cls=KernelNode, bin_tree_cls=KernelTree)


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
        # consider kernels equal if they have the same active dimension and class
        nodes_eq = same_dims and same_cls
        if nodes_eq:
            return 0
        else:
            return 1
    else:
        return 1

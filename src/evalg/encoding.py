from typing import List, Optional, TypeVar

from GPy.kern.src.kern import Kern
from graphviz import Digraph

import src.autoks.kernel

operators = ['+', '*']
T = TypeVar('T')


def val_to_label(value: T) -> str:
    """Convert value to a string a label.

    :param value:
    :return:
    """
    if isinstance(value, Kern):
        return src.autoks.kernel.subkernel_expression(value)
    else:
        return str(value)


class TreeNode:
    label: str
    parent: Optional
    _value: T

    def __init__(self, value: T, parent=None):
        self._value = value
        self.label = val_to_label(value)
        self.parent = parent

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = value
        # update label as well
        self.label = val_to_label(value)

    def __str__(self):
        return self.label

    def __repr__(self):
        return f'{self.__class__.__name__}('f'label={self.label!r}, value={self.value!r}, parent={self.parent!r})'


class BinaryTreeNode(TreeNode):

    def __init__(self, value, parent=None):
        super().__init__(value, parent)

        self.left: Optional = None
        self.right: Optional = None

    def add_left(self, value: T):
        """Add left node.

        :param value:
        :return:
        """
        self.left = BinaryTreeNode(value, self)
        return self.left

    def add_right(self, value: T):
        """Add right node.

        :param value:
        :return:
        """
        self.right = BinaryTreeNode(value, self)
        return self.right

    def create_graph(self, graph: Optional[Digraph] = None) -> Digraph:
        """Create a graphviz graph of the binary tree node.

        :param graph:
        :return:
        """
        root = self
        if not graph:
            graph = Digraph()

        if root is not None:
            root_id = str(id(root))
            graph.node(root_id, label=root.label)

            if root.left is not None:
                left = root.left
                left_id = str(id(left))
                graph.node(left_id, label=left.label)
                graph.edge(root_id, left_id)
                root.left.create_graph(graph=graph)
            if root.right is not None:
                right = root.right
                right_id = str(id(right))
                graph.node(right_id, label=right.label)
                graph.edge(root_id, right_id)
                root.right.create_graph(graph=graph)

        return graph

    def __str__(self):
        return self.label

    def __repr__(self):
        return f'{self.__class__.__name__}('f'label={self.label!r}, value={self.value!r}, parent={self.parent!r}, ' \
            f'left={self.left!r}, right={self.right!r})'


class BinaryTree:
    root: Optional[BinaryTreeNode]

    def __init__(self, root: BinaryTreeNode = None):
        if root is not None and not isinstance(root, BinaryTreeNode):
            raise TypeError('root must be a {}'.format(BinaryTreeNode.__name__))
        self.root = root

    def create_graph(self) -> Digraph:
        """Create a graphviz graph of the binary tree.

        :return:
        """
        return self.root.create_graph()

    def select_postorder(self, node_idx: int) -> Optional[BinaryTreeNode]:
        """Select node from binary tree given postorder index.

        :param node_idx:
        :return:
        """
        node = self.root
        stack = []
        last_node_visited = None
        i = 0
        while len(stack) > 0 or node is not None:
            if node:
                stack.append(node)
                node = node.left
            else:
                peek_node = stack[-1]
                if peek_node.right is not None and last_node_visited is not peek_node.right:
                    node = peek_node.right
                else:
                    if i == node_idx:
                        return peek_node
                    last_node_visited = stack.pop()
                    i += 1

        return None

    def infix(self) -> str:
        """In-order string representation of the binary tree.

        :return:
        """
        return self._infix_helper(self.root)

    def height(self) -> int:
        """Height of the tree.

        :return:
        """
        return self._height_helper(self.root)

    def infix_tokens(self) -> list:
        """Infix tokens of the binary tree.

        :return:
        """
        return self._infix_tokens_helper(self.root)

    def postfix_tokens(self) -> list:
        """Postfix tokens of the binary tree.

        :return:
        """
        return infix_tokens_to_postfix_tokens(self.infix_tokens())

    def _infix_helper(self, root: BinaryTreeNode, expression: Optional[str] = None) -> str:
        """Helper function to get the infix string of a binary tree node.

        :param root:
        :param expression:
        :return:
        """
        if expression is None:
            expression = ''

        if root is not None:
            if root.value in operators:
                expression += '('

            expression += self._infix_helper(root.left)
            expression += root.label
            expression += self._infix_helper(root.right)

            if root.value in operators:
                expression += ')'

        return expression

    def _infix_tokens_helper(self, root: BinaryTreeNode, tokens: Optional[list] = None) -> list:
        """Helper function to get the infix tokens of a binary tree node.

        :param root:
        :param tokens:
        :return:
        """
        if tokens is None:
            tokens = []

        if root is not None:
            if root.value in operators:
                tokens += ['(']

            tokens += self._infix_tokens_helper(root.left)
            tokens += [root.label]
            tokens += self._infix_tokens_helper(root.right)

            if root.value in operators:
                tokens += [')']

        return tokens

    def _height_helper(self, node: Optional[BinaryTreeNode]) -> int:
        """Helper function to get the height of a binary tree node.

        :param node:
        :return:
        """
        if node is None:
            return 0

        # Get the depth of each sub-tree
        left_depth = self._height_helper(node.left)
        right_depth = self._height_helper(node.right)

        if left_depth > right_depth:
            return left_depth + 1
        else:
            return right_depth + 1

    def __repr__(self):
        return f'{self.__class__.__name__}('f'root={self.root!r})'


def postfix_tokens_to_binexp_tree(postfix_tokens: List[str]) -> BinaryTree:
    """Convert postfix tokens to a binary tree.

    :param postfix_tokens:
    :return:
    """
    tree = BinaryTree()

    root = BinaryTreeNode(postfix_tokens[-1])
    tree.root = root

    curr = root
    for token in postfix_tokens[-2::-1]:
        # while curr can't have more children
        while curr.value not in operators or (curr.right is not None and curr.left is not None):
            curr = curr.parent

        if curr.right is None:
            node = curr.add_right(token)
        elif curr.left is None:
            node = curr.add_left(token)
        curr = node

    return tree


def infix_tokens_to_postfix_tokens(infix_tokens: List[str]) -> list:
    """Convert infix tokens to postfix tokens.

    :param infix_tokens:
    :return:
    """
    pemdas = {}
    pemdas["*"] = 3
    pemdas["+"] = 1
    pemdas["("] = 0

    operator_stack = []
    postfix_tokens = []
    for token in infix_tokens:
        if token in operators:
            while len(operator_stack) is not 0 and pemdas[operator_stack[-1]] >= pemdas[token]:
                postfix_tokens.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':  # nosec
            operator_stack.append(token)
        elif token == ')':  # nosec
            top_token = operator_stack.pop()
            while top_token != '(':
                postfix_tokens.append(top_token)
                top_token = operator_stack.pop()
        else:
            # token is an operand
            postfix_tokens.append(token)

    while len(operator_stack) > 0:
        postfix_tokens.append(operator_stack.pop())

    return postfix_tokens

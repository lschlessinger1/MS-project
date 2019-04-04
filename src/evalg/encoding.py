from typing import List, Optional, TypeVar, Sequence, Type

from graphviz import Digraph

operators = ['+', '*']
T = TypeVar('T')


class TreeNode:
    label: str
    html_label: str
    parent: Optional
    children: Optional[Sequence]
    _value: T

    def __init__(self,
                 value: T,
                 parent=None,
                 children=None):
        self.value = value
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children

    def _value_to_label(self, value) -> str:
        """Convert value to a string a label.

        :param value:
        :return:
        """
        return str(value)

    def _value_to_html(self, value) -> str:
        return '<' + self._value_to_label(value) + '>'

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = value
        # update label as well
        self.label = self._value_to_label(value)
        self.html_label = self._value_to_html(value)

    def has_parent(self) -> bool:
        return self.parent is not None

    def has_children(self) -> bool:
        return self.children is not None and len(self.children) > 0

    def add_child(self, value: T):
        """Add child node.

        :param value:
        :return:
        """
        child = self.__class__(value=value, parent=self)
        self._add_child(child)
        return child

    def _add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return self.label

    def __repr__(self):
        parent_label = None if not self.parent else self.parent.label
        return f'{self.__class__.__name__}('f'label={self.label!r}, value={self.value!r}, parent={parent_label!r})'


class BinaryTreeNode(TreeNode):
    left: Optional
    right: Optional

    def __init__(self, value, parent=None, left=None, right=None):
        children = [left, right]
        super().__init__(value, parent, children)

        self.left = left
        self.right = right

    def has_left_child(self) -> bool:
        return self.left is not None

    def has_right_child(self) -> bool:
        return self.right is not None

    def is_left_child(self) -> bool:
        if self.has_parent():
            if self.parent.has_left_child():
                return self.parent.left == self
        else:
            raise AttributeError(f'{self.__class__.__name__!r} does not have a parent.')

    def is_right_child(self) -> bool:
        if self.has_parent():
            if self.parent.has_right_child():
                return self.parent.right == self
        else:
            raise AttributeError(f'{self.__class__.__name__!r} does not have a parent.')

    def is_root(self) -> bool:
        return not self.parent

    def is_leaf(self) -> bool:
        return not (self.right or self.left)

    def add_left(self, value: T):
        """Add left node.

        :param value:
        :return:
        """
        self.left = self.__class__(value, self)
        self._add_child(self.right)
        return self.left

    def add_right(self, value: T):
        """Add right node.

        :param value:
        :return:
        """
        self.right = self.__class__(value, self)
        self._add_child(self.right)
        return self.right

    def create_graph(self, graph: Optional[Digraph] = None, **digraph_kwargs) -> Digraph:
        """Create a graphviz graph of the binary tree node.

        :param graph:
        :return:
        """
        root = self
        if not graph:
            default_name = 'binary_tree_node'
            if 'name' not in digraph_kwargs:
                digraph_kwargs['name'] = default_name
            graph = Digraph(**digraph_kwargs)

        if root is not None:
            leaf_shape = 'box'

            root_shape = leaf_shape if self.is_leaf() else None
            root_id = str(id(root))
            graph.node(root_id, label=root.html_label, shape=root_shape)

            if root.left is not None:
                left = root.left
                left_shape = leaf_shape if left.is_leaf() else None
                left_id = str(id(left))
                graph.node(left_id, label=left.html_label, shape=left_shape)
                graph.edge(root_id, left_id)
                root.left.create_graph(graph=graph)
            if root.right is not None:
                right = root.right
                right_shape = leaf_shape if right.is_leaf() else None
                right_id = str(id(right))
                graph.node(right_id, label=right.html_label, shape=right_shape)
                graph.edge(root_id, right_id)
                root.right.create_graph(graph=graph)

        return graph

    def height(self) -> int:
        """Return the height of the this node

        The height of a node is the number of edges on the longest path between that node and a descendant leaf.

        :return:
        """
        return self._height_helper(self)

    def _height_helper(self, node: Optional) -> int:
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

    def __iter__(self):
        """Pre-order traversal of the binary tree node."""
        if self:
            yield self.value
            if self.has_left_child():
                for item in self.left:
                    yield item
            if self.has_right_child():
                for item in self.right:
                    yield item

    def __contains__(self, value):
        """Returns True if node is in some node of the tree, False otherwise."""
        for item in self:
            if value == item:
                return True
        return False

    def __len__(self):
        return self._size(self)

    def _size(self, node):
        n = 1
        if node.has_left_child():
            n += self._size(node.left)
        if node.has_right_child():
            n += self._size(node.right)
        return n

    def __str__(self):
        return self.label

    def __repr__(self):
        parent_label = None if not self.parent else self.parent.label
        left_label = None if not self.left else self.left.label
        right_label = None if not self.right else self.right.label
        return f'{self.__class__.__name__}('f'label={self.label!r}, value={self.value!r}, left={left_label!r}, ' \
            f'right={right_label!r}, parent={parent_label!r})'


class BinaryTree:
    _root: Optional[BinaryTreeNode]

    def __init__(self, root: Optional[BinaryTreeNode] = None):
        self._root = root

    @property
    def root(self) -> Optional[BinaryTreeNode]:
        return self._root

    @root.setter
    def root(self, root: Optional[BinaryTreeNode]) -> None:
        if root is not None and not isinstance(root, BinaryTreeNode):
            raise TypeError('root must be a {}'.format(BinaryTreeNode.__name__))
        self._root = root

    def create_graph(self, **digraph_kwargs) -> Digraph:
        """Create a graphviz graph of the binary tree.

        :return:
        """
        default_name = 'binary_tree'
        if 'name' not in digraph_kwargs:
            digraph_kwargs['name'] = default_name
        return self.root.create_graph(**digraph_kwargs)

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
        """Height of the tree (height of its root node).

        :return:
        """
        return 0 if not self.root else self.root.height()

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

    def _infix_helper(self,
                      root: BinaryTreeNode,
                      expression: Optional[str] = None) -> str:
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

    def _infix_tokens_helper(self,
                             root: BinaryTreeNode,
                             tokens: Optional[list] = None) -> list:
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

    def __iter__(self):
        return self.root.__iter__()

    def __contains__(self, value):
        return self.root.__contains__(value)

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return f'{self.__class__.__name__}('f'root={self.root!r})'


def postfix_tokens_to_bin_tree_node(postfix_tokens: List[str],
                                    bin_tree_cls: Type[BinaryTreeNode] = BinaryTreeNode) -> BinaryTreeNode:
    """Convert postfix tokens to a binary tree node.

    :param postfix_tokens:
    :param bin_tree_cls: Binary tree node class
    :return:
    """
    root = bin_tree_cls(postfix_tokens[-1])

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

    return root


V = TypeVar('V')


def postfix_tokens_to_binexp_tree(postfix_tokens: List[str],
                                  bin_tree_node_cls: Type[BinaryTreeNode] = BinaryTreeNode,
                                  bin_tree_cls: Type[V] = BinaryTree) -> V:
    root = postfix_tokens_to_bin_tree_node(postfix_tokens, bin_tree_node_cls)
    return bin_tree_cls(root)


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

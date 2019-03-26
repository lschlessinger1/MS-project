from unittest import TestCase

from GPy.kern import RBF
from graphviz import Digraph

from src.evalg.encoding import TreeNode, BinaryTreeNode, BinaryTree, infix_tokens_to_postfix_tokens, \
    postfix_tokens_to_binexp_tree


class TestTreeNode(TestCase):

    def setUp(self):
        self.parent_val = 'Parent Value'
        self.parent_label = self.parent_val
        self.parent = TreeNode(self.parent_val, parent=None)

        self.child_val = 42
        self.child_label = str(self.child_val)
        self.child = self.parent.add_child(self.child_val)
        # self.child = TreeNode(self.child_val, parent=self.parent)

        self.kernel_node_val_prev = 10
        self.kernel_node_label_prev = str(self.kernel_node_val_prev)
        self.kernel_node = TreeNode(self.kernel_node_val_prev)

    def test_value(self):
        # test value
        self.assertEqual(self.child.value, self.child_val)
        self.assertEqual(self.parent.value, self.parent_val)
        self.assertEqual(self.kernel_node.value, self.kernel_node_val_prev)

        new_val = RBF(1, [0])
        new_label = 'SE0'
        self.kernel_node.value = new_val

        self.assertEqual(self.kernel_node.value, new_val)

        # test label
        self.assertEqual(self.parent.label, self.parent_label)
        self.assertEqual(self.child.label, self.child_label)
        self.assertEqual(self.kernel_node.label, new_label)

    def test_has_parent(self):
        self.assertFalse(self.parent.has_parent())
        self.assertTrue(self.child.has_parent())

    def test_has_children(self):
        self.assertTrue(self.parent.has_children())
        self.assertFalse(self.child.has_children())

    def test_add_child(self):
        node = TreeNode(self.parent_val)
        result = node.add_child('value')
        self.assertIsInstance(result, TreeNode)
        self.assertTrue(result.has_parent())
        self.assertTrue(node.has_children())
        self.assertEqual(node.children[0], result)
        self.assertEqual(result.parent, node)


class TestBinaryTreeNode(TestCase):

    def setUp(self):
        self.root_val = 'Parent Value'
        self.root = BinaryTreeNode(self.root_val)

        self.left_child_val = 42
        self.right_child_val = 13

    def test_has_left_child(self):
        self.assertFalse(self.root.has_left_child())
        self.root.add_right(self.right_child_val)
        self.assertFalse(self.root.has_left_child())
        self.root.add_left(self.left_child_val)
        self.assertTrue(self.root.has_left_child())

    def test_has_right_child(self):
        self.assertFalse(self.root.has_right_child())
        self.root.add_left(self.left_child_val)
        self.assertFalse(self.root.has_right_child())
        self.root.add_right(self.right_child_val)
        self.assertTrue(self.root.has_right_child())

    def test_has_parent(self):
        self.assertFalse(self.root.has_parent())
        right = self.root.add_right(self.right_child_val)
        self.assertTrue(right.has_parent())
        self.assertTrue(self.root.right.has_parent())
        self.assertFalse(self.root.has_parent())
        left = self.root.add_left(self.left_child_val)
        self.assertTrue(left.has_parent())
        self.assertTrue(self.root.left.has_parent())
        self.assertFalse(self.root.has_parent())

    def test_is_left_child(self):
        self.assertRaises(AttributeError, self.root.is_left_child)
        left = self.root.add_left(self.left_child_val)
        self.assertTrue(left.is_left_child())
        self.assertTrue(self.root.left.is_left_child())
        right = self.root.add_right(self.right_child_val)
        self.assertFalse(right.is_left_child())
        self.assertFalse(self.root.right.is_left_child())

    def test_is_right_child(self):
        self.assertRaises(AttributeError, self.root.is_right_child)
        left = self.root.add_left(self.left_child_val)
        self.assertFalse(left.is_right_child())
        self.assertFalse(self.root.left.is_right_child())
        right = self.root.add_right(self.right_child_val)
        self.assertTrue(right.is_right_child())
        self.assertTrue(self.root.right.is_right_child())

    def test_is_root(self):
        self.assertTrue(self.root.is_root())
        left = self.root.add_left(self.left_child_val)
        self.assertFalse(left.is_root())
        right = self.root.add_right(self.right_child_val)
        self.assertFalse(right.is_root())

    def test_is_leaf(self):
        self.assertTrue(self.root.is_leaf())
        left = self.root.add_left(self.left_child_val)
        self.assertTrue(left.is_leaf())
        self.assertFalse(self.root.is_leaf())
        right = self.root.add_right(self.right_child_val)
        self.assertTrue(right.is_leaf())
        self.assertFalse(self.root.is_leaf())

    def test_add_left(self):
        result = self.root.add_left(self.left_child_val)
        self.assertEqual(result.parent, self.root)
        self.assertEqual(result.parent.value, self.root_val)
        self.assertEqual(result.parent.left, result)
        self.assertEqual(result.parent.left.value, self.left_child_val)

    def test_add_right(self):
        result = self.root.add_right(self.right_child_val)
        self.assertEqual(result.parent, self.root)
        self.assertEqual(result.parent.value, self.root_val)
        self.assertEqual(result.parent.right, result)
        self.assertEqual(result.parent.right.value, self.right_child_val)

    def test_create_graph(self):
        result = self.root.create_graph()
        self.assertIsInstance(result, Digraph)

    def test_height(self):
        root = BinaryTreeNode('*')
        self.assertEqual(root.height(), 1)

        left = root.add_left(10)
        self.assertEqual(root.height(), 2)
        right = root.add_right(20)
        self.assertEqual(root.height(), 2)

        ll = left.add_left(40)
        self.assertEqual(root.height(), 3)
        left.add_right(50)
        self.assertEqual(root.height(), 3)
        right.add_left(60)
        self.assertEqual(root.height(), 3)
        right.add_right(70)
        self.assertEqual(root.height(), 3)

        ll.add_left(80)
        self.assertEqual(root.height(), 4)


class TestBinaryTree(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode('*')
        self.tree.root = self.root

    def test_root(self):
        tree = BinaryTree()
        with self.assertRaises(TypeError):
            tree.root = 'bad type'

    def test_create_graph(self):
        result = self.tree.create_graph()
        self.assertIsInstance(result, Digraph)

    def test_select_postorder(self):
        left = self.root.add_left(20)
        right = self.root.add_right(30)
        ll = left.add_left(40)
        lr = left.add_right(50)
        rl = right.add_left(60)
        rr = right.add_right(70)
        self.assertEqual(self.tree.select_postorder(0), ll)
        self.assertEqual(self.tree.select_postorder(1), lr)
        self.assertEqual(self.tree.select_postorder(2), left)
        self.assertEqual(self.tree.select_postorder(3), rl)
        self.assertEqual(self.tree.select_postorder(4), rr)
        self.assertEqual(self.tree.select_postorder(5), right)
        self.assertEqual(self.tree.select_postorder(6), self.root)

    def test_height(self):
        tree = BinaryTree()
        self.assertEqual(tree.height(), 0)

        tree.root = BinaryTreeNode('*')
        self.assertEqual(tree.height(), 1)

        left = tree.root.add_left(10)
        self.assertEqual(tree.height(), 2)
        right = tree.root.add_right(20)
        self.assertEqual(tree.height(), 2)

        ll = left.add_left(40)
        self.assertEqual(tree.height(), 3)
        left.add_right(50)
        self.assertEqual(tree.height(), 3)
        right.add_left(60)
        self.assertEqual(tree.height(), 3)
        right.add_right(70)
        self.assertEqual(tree.height(), 3)

        ll.add_left(80)
        self.assertEqual(tree.height(), 4)

    def test_infix_tokens(self):
        left = self.root.add_left('+')
        right = self.root.add_right('+')
        left.add_left('A')
        left.add_right('B')
        right.add_left('C')
        right.add_right('D')

        tokens = ['(', '(', 'A', '+', 'B', ')', self.tree.root.label, '(', 'C', '+', 'D', ')', ')']
        result = self.tree.infix_tokens()
        self.assertCountEqual(result, tokens)

    def test_postfix_tokens(self):
        tree = BinaryTree()
        root = BinaryTreeNode('*')
        tree.root = root

        left = root.add_left('+')
        right = root.add_right('+')
        left.add_left('A')
        left.add_right('B')
        right.add_left('C')
        right.add_right('D')

        tokens = ['A', 'B', 'C', tree.root.label, '+', 'D', '+']
        result = tree.postfix_tokens()
        self.assertCountEqual(result, tokens)

        tree = BinaryTree()
        root = BinaryTreeNode('+')
        tree.root = root

        left = root.add_left('+')
        right = root.add_right('+')
        left.add_left('A')
        left.add_right('B')
        right.add_left('C')
        right.add_right('D')

        tokens = ['A', 'B', '+', 'C', '+', 'D', '+']
        result = tree.postfix_tokens()
        self.assertCountEqual(result, tokens)


class TestEncoding(TestCase):

    def test_infix_tokens_to_postfix_tokens(self):
        infix_tokens = ['A', '+', 'B', '*', 'C', '+', 'D']
        result = infix_tokens_to_postfix_tokens(infix_tokens)
        postfix_tokens = ['A', 'B', 'C', '*', '+', 'D', '+']
        self.assertCountEqual(result, postfix_tokens)

        infix_tokens = ['(', 'A', '+', 'B', ')', '*', '(', 'C', '+', 'D', ')']
        result = infix_tokens_to_postfix_tokens(infix_tokens)
        postfix_tokens = ['A', 'B', 'C', '*', '+', 'D', '+']
        self.assertCountEqual(result, postfix_tokens)

        infix_tokens = ['A', '*', 'B', '+', 'C', '*', 'D']
        result = infix_tokens_to_postfix_tokens(infix_tokens)
        postfix_tokens = ['A', 'B', '*', 'C', 'D', '*', '+']
        self.assertCountEqual(result, postfix_tokens)

        infix_tokens = ['(', '(', 'A', '+', 'B', ')', '+', '(', 'C', '+', 'D', ')', ')']
        result = infix_tokens_to_postfix_tokens(infix_tokens)
        postfix_tokens = ['A', 'B', '+', 'C', '+', 'D', '+']
        self.assertCountEqual(result, postfix_tokens)

        infix_tokens = ['A', '+', 'B', '+', 'C', '+', 'D']
        result = infix_tokens_to_postfix_tokens(infix_tokens)
        postfix_tokens = ['A', 'B', '+', 'C', '+', 'D', '+']
        self.assertCountEqual(result, postfix_tokens)

    def test_postfix_tokens_to_binexp_tree(self):
        # (A + B) * (C + D)
        postfix_tokens = ['A', 'B', '+', 'C', 'D', '+', '*']
        tree = postfix_tokens_to_binexp_tree(postfix_tokens)
        root = tree.root
        self.assertEqual(root.label, '*')
        self.assertEqual(root.left.label, '+')
        self.assertEqual(root.right.label, '+')
        self.assertEqual(root.left.left.label, 'A')
        self.assertEqual(root.left.right.label, 'B')
        self.assertEqual(root.right.left.label, 'C')
        self.assertEqual(root.right.right.label, 'D')

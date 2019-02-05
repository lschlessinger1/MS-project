from unittest import TestCase

from GPy.kern import RBF
from graphviz import Digraph

from evalg.encoding import TreeNode, BinaryTreeNode, BinaryTree


class TestTreeNode(TestCase):

    def setUp(self):
        self.parent_val = 'Parent Value'
        self.parent_label = self.parent_val
        self.parent = TreeNode(self.parent_val, parent=None)

        self.child_val = 42
        self.child_label = str(self.child_val)
        self.child = TreeNode(self.child_val, parent=self.parent)

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


class TestBinaryTreeNode(TestCase):

    def setUp(self):
        self.root_val = 'Parent Value'
        self.root = BinaryTreeNode(self.root_val)

        self.left_child_val = 42
        self.right_child_val = 13

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


class TestBinaryTree(TestCase):

    def setUp(self):
        self.tree = BinaryTree()
        self.root = BinaryTreeNode(10)
        self.tree.root = self.root

    def test_create_graph(self):
        result = self.tree.create_graph()
        self.assertIsInstance(result, Digraph)

    def test_select_postorder(self):
        l = self.root.add_left(20)
        r = self.root.add_right(30)
        ll = l.add_left(40)
        lr = l.add_right(50)
        rl = r.add_left(60)
        rr = r.add_right(70)
        self.assertEqual(self.tree.select_postorder(0), ll)
        self.assertEqual(self.tree.select_postorder(1), lr)
        self.assertEqual(self.tree.select_postorder(2), l)
        self.assertEqual(self.tree.select_postorder(3), rl)
        self.assertEqual(self.tree.select_postorder(4), rr)
        self.assertEqual(self.tree.select_postorder(5), r)
        self.assertEqual(self.tree.select_postorder(6), self.root)

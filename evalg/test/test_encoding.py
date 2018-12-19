from unittest import TestCase

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

    def test_get_parent(self):
        self.assertEqual(self.child.get_parent(), self.parent)

    def test_get_value(self):
        self.assertEqual(self.child.get_value(), self.child_val)
        self.assertEqual(self.parent.get_value(), self.parent_val)

    def test_get_label(self):
        self.assertEqual(self.parent.get_label(), str(self.parent_label))
        self.assertEqual(self.child.get_label(), str(self.child_label))


class TestBinaryTreeNode(TestCase):

    def setUp(self):
        self.root_val = 'Parent Value'
        self.root = BinaryTreeNode(self.root_val)

        self.left_child_val = 42
        self.right_child_val = 13

    def test_add_left(self):
        result = self.root.add_left(self.left_child_val)
        self.assertEqual(result.get_parent(), self.root)
        self.assertEqual(result.get_parent().get_value(), self.root_val)
        self.assertEqual(result.get_parent().left, result)
        self.assertEqual(result.get_parent().left.get_value(), self.left_child_val)

    def test_add_right(self):
        result = self.root.add_right(self.right_child_val)
        self.assertEqual(result.get_parent(), self.root)
        self.assertEqual(result.get_parent().get_value(), self.root_val)
        self.assertEqual(result.get_parent().right, result)
        self.assertEqual(result.get_parent().right.get_value(), self.right_child_val)

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

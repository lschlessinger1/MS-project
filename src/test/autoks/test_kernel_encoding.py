from unittest import TestCase

from GPy.kern import RBF, RationalQuadratic

from src.autoks.core.kernel_encoding import KernelNode, KernelTree, hd_kern_nodes


class TestKernelNode(TestCase):

    def test_init(self):
        kern = RBF(1)
        result = KernelNode(kern)
        self.assertEqual(result.value, kern)
        self.assertEqual(result.label, 'SE_0')
        self.assertIsNone(result.parent)
        self.assertIsNone(result.left)
        self.assertIsNone(result.right)

    def test__value_to_label(self):
        mock_kern = RBF(1)
        node = KernelNode(mock_kern)
        result = node._value_to_label(mock_kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'SE_0')


class TestKernelTree(TestCase):

    def test_init(self):
        kern = RBF(1)
        root = KernelNode(kern)
        result = KernelTree(root)
        self.assertEqual(result.root, root)
        self.assertEqual(result.root.label, 'SE_0')
        self.assertEqual(result.root.value, kern)


class TestKernelEncoding(TestCase):

    def test_hd_kern_nodes(self):
        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_2 = KernelNode(RBF(1, active_dims=[0]))
        result = hd_kern_nodes(node_1, node_2)
        self.assertEqual(result, 0)

        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_2 = KernelNode(RationalQuadratic(1, active_dims=[0]))
        result = hd_kern_nodes(node_1, node_2)
        self.assertEqual(result, 1)

        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_2 = KernelNode(RBF(1, active_dims=[1]))
        result = hd_kern_nodes(node_1, node_2)
        self.assertEqual(result, 1)

        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_1.add_left('U')
        node_1.add_right('V')
        node_2 = KernelNode(RBF(1, active_dims=[0]))
        result = hd_kern_nodes(node_1, node_2)
        self.assertEqual(result, 1)

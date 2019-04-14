import unittest
from unittest.mock import MagicMock

import numpy as np
from GPy.kern import RBF, Add, RatQuad, Prod, KernelKernel

from src.autoks.kernel import sort_kernel, AKSKernel, get_all_1d_kernels, create_1d_kernel, \
    remove_duplicate_aks_kernels, set_priors, KernelTree, KernelNode, subkernel_expression, shd_metric, \
    decode_kernel, hd_kern_nodes, encode_kernel, encode_aks_kerns, shd_kernel_kernel, encode_aks_kernel, \
    euclidean_metric, euclidean_kernel_kernel
from src.autoks.util import remove_duplicates
from src.test.autoks.support.util import has_combo_kernel_type


class TestKernel(unittest.TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

    def test_sort_kernel(self):
        k = self.se0
        result = sort_kernel(k)
        # should be SE0
        self.assertEqual(result.active_dims[0], 0)
        self.assertIsInstance(result, RBF)

        k = self.se1 + self.se0
        result = sort_kernel(k)
        # should be SE0 + SE1
        self.assertIsInstance(result, Add)
        self.assertIsInstance(result.parts[0], RBF)
        self.assertIsInstance(result.parts[1], RBF)
        self.assertEqual(result.parts[0].active_dims[0], 0)
        self.assertEqual(result.parts[1].active_dims[0], 1)

        k = self.se1 + self.se0 + self.rq1 + self.rq0
        result = sort_kernel(k)
        # should be RQ0 + RQ1 + SE0 + SE1
        self.assertIsInstance(result, Add)
        kernel_types = [(RatQuad, 0), (RatQuad, 1), (RBF, 0), (RBF, 1)]
        for (k_class, dim), part in zip(kernel_types, result.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        k = self.rq1 * self.rq0 + self.se1 * self.se0 * self.rq1 + self.se1 * self.rq0 * self.rq1
        result = sort_kernel(k)
        # should be (RQ1 * SE0 * SE1) + (RQ0 * RQ1) + (RQ0 * RQ1 * SE1)
        self.assertIsInstance(result, Add)
        kernel_types_outer = [Prod, Prod, Prod]
        for k_class, part in zip(kernel_types_outer, result.parts):
            self.assertIsInstance(part, k_class)

        kernel_types_inner_1 = [(RatQuad, 1), (RBF, 0), (RBF, 1)]
        prod_1 = result.parts[0]
        for (k_class, dim), part in zip(kernel_types_inner_1, prod_1.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_2 = [(RatQuad, 0), (RatQuad, 1)]
        prod_2 = result.parts[1]
        for (k_class, dim), part in zip(kernel_types_inner_2, prod_2.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_3 = [(RatQuad, 0), (RatQuad, 1), (RBF, 1)]
        prod_3 = result.parts[2]
        for (k_class, dim), part in zip(kernel_types_inner_3, prod_3.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

    def test_remove_duplicates(self):
        # simple example
        data = [1, 1, 1, 2, 3, 4, 'a', 'b', True]
        values = [10, 9, 8, '7', '6', False, 4, 3, 2]
        result = remove_duplicates(data, values)
        self.assertEqual(result, [10, '7', '6', False, 4, 3])

        with self.assertRaises(ValueError):
            remove_duplicates([1, 2, 3], ['1', 2])

    def test_remove_duplicate_aks_kernels(self):
        k1 = AKSKernel(RBF(1))
        k1.score = 10

        k2 = AKSKernel(RBF(1))
        k2.score = 9

        k3 = AKSKernel(RBF(1))
        k3.nan_scored = True

        k4 = AKSKernel(RBF(1))
        k4.nan_scored = True

        k5 = AKSKernel(RBF(1))

        k6 = AKSKernel(RBF(1, lengthscale=0.5))

        k7 = AKSKernel(RatQuad(1))

        # Always keep k1 then k2 then k3 etc.
        result = remove_duplicate_aks_kernels([k1, k2, k3, k4, k5, k6, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k1, k2, k3, k4, k5, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k1, k2, k3, k4, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k1, k2, k3, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k1, k2, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k1, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_aks_kernels([k2, k3, k4, k5, k6, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_aks_kernels([k2, k3, k4, k5, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_aks_kernels([k2, k3, k4, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_aks_kernels([k2, k3, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_aks_kernels([k2, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_aks_kernels([k3, k4, k5, k6, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k3, k4, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k4, k3, k5, k6, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k4, k3, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k3, k4, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k3, k4, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k4, k3, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_aks_kernels([k3, k7])
        self.assertListEqual(result, [k3, k7])

        result = remove_duplicate_aks_kernels([k4, k7])
        self.assertListEqual(result, [k4, k7])

        result = remove_duplicate_aks_kernels([k5, k6, k7])
        self.assertTrue(result == [k5, k7] or result == [k6, k7])

        result = remove_duplicate_aks_kernels([k6, k5, k7])
        self.assertTrue(result == [k5, k7] or result == [k6, k7])

    def test_get_all_1d_kernels(self):
        result = get_all_1d_kernels(['SE', 'RQ'], n_dims=2)
        self.assertIsInstance(result, list)
        kernel_types = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertEqual(len(result), len(kernel_types))
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        result = get_all_1d_kernels(['SE', 'RQ'], n_dims=1)
        self.assertIsInstance(result, list)
        kernel_types = [self.se0, self.rq0]
        self.assertEqual(len(result), len(kernel_types))
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

    def test_create_1d_kernel(self):
        result = create_1d_kernel(active_dim=0, kernel_family='SE')
        self.assertIsInstance(result, RBF)
        self.assertEqual(result.active_dims[0], 0)

        result = create_1d_kernel(active_dim=1, kernel_family='RQ')
        self.assertIsInstance(result, RatQuad)
        self.assertEqual(result.active_dims[0], 1)

    def test_set_priors(self):
        priors = dict()
        param = RBF(1)
        mock_prior = MagicMock()
        priors['variance'] = mock_prior
        result = set_priors(param, priors)
        self.assertEqual(result['variance'].priors.properties()[0], mock_prior)

    def test_subkernel_expression(self):
        kernel = RBF(1, variance=3, lengthscale=2)
        result = subkernel_expression(kernel=kernel, show_params=False, html_like=False)
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'SE_0')
        result = subkernel_expression(kernel=kernel, show_params=True, html_like=False)
        self.assertIsInstance(result, str)
        self.assertIn('SE_0', result)
        self.assertIn('variance', result)
        self.assertIn('lengthscale', result)
        self.assertIn('3', result)
        self.assertIn('2', result)
        result = subkernel_expression(kernel=kernel, show_params=False, html_like=True)
        self.assertIsInstance(result, str)
        self.assertEqual(result, '<SE<SUB><FONT POINT-SIZE="8">0</FONT></SUB>>')
        result = subkernel_expression(kernel=kernel, show_params=True, html_like=True)
        self.assertIsInstance(result, str)
        self.assertIn('<SE<SUB><FONT POINT-SIZE="8">0</FONT></SUB>>', result)
        self.assertIn('variance', result)
        self.assertIn('lengthscale', result)
        self.assertIn('3', result)
        self.assertIn('2', result)

    def test_shd_metric(self):
        aks_kernels = [AKSKernel(RBF(1) + RatQuad(1)), AKSKernel(RBF(1))]
        data = encode_aks_kerns(aks_kernels)
        u, v = data[0], data[1]
        result = shd_metric(u, v)
        self.assertEqual(result, 1)

    def test_decode_kernel(self):
        kern = RBF(1)
        kern_dict = kern.to_dict()
        kern_dict_str = str(kern_dict)
        result = decode_kernel(kern_dict_str)
        self.assertIsInstance(result, RBF)
        self.assertEqual(result.input_dim, 1)
        self.assertDictEqual(result.to_dict(), kern_dict)

    def test_hd_kern_nodes(self):
        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_2 = KernelNode(RBF(1, active_dims=[0]))
        result = hd_kern_nodes(node_1, node_2)
        self.assertEqual(result, 0)

        node_1 = KernelNode(RBF(1, active_dims=[0]))
        node_2 = KernelNode(RatQuad(1, active_dims=[0]))
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

    def test_encode_kernel(self):
        kern = RBF(1, active_dims=[0])
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

        kern = RBF(1, active_dims=[0]) + RBF(1)
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

    def test_encode_aks_kernel(self):
        aks_kernel = AKSKernel(RBF(1, active_dims=[0]))
        result = encode_aks_kernel(aks_kernel)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], str)
        self.assertListEqual(result, [encode_kernel(aks_kernel.kernel)])

    def test_encode_aks_kerns(self):
        aks_kernels = [AKSKernel(RBF(1)), AKSKernel(RatQuad(1))]
        result = encode_aks_kerns(aks_kernels)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(aks_kernels), 1))
        self.assertListEqual(result[0][0], [encode_kernel(aks_kernels[0].kernel)])
        self.assertListEqual(result[1][0], [encode_kernel(aks_kernels[1].kernel)])

        aks_kernels = [AKSKernel(RBF(1) * RBF(1)), AKSKernel(RatQuad(1))]
        result = encode_aks_kerns(aks_kernels)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(aks_kernels), 1))
        self.assertListEqual(result[0][0], [encode_kernel(aks_kernels[0].kernel)])
        self.assertListEqual(result[1][0], [encode_kernel(aks_kernels[1].kernel)])

    def test_shd_kernel_kernel(self):
        result = shd_kernel_kernel(variance=1, lengthscale=1)
        self.assertIsInstance(result, KernelKernel)
        self.assertEqual(result.dist_metric, shd_metric)

        aks_kernels_evaluated = [AKSKernel(RBF(1)), AKSKernel(RatQuad(1))]
        x = encode_aks_kerns(aks_kernels_evaluated)
        d = result._unscaled_dist(x)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue(np.array_equal(d, np.array([[0, 1], [1, 0]])))
        k = RBF(1, variance=1, lengthscale=1)
        self.assertTrue(np.array_equal(result.K(x), k.K(d)))

    def test_euclidean_metric(self):
        x_train = np.array([[1, 2], [3, 4]])
        aks_kernels = [AKSKernel(RBF(1) + RatQuad(1)), AKSKernel(RBF(1))]
        data = encode_aks_kerns(aks_kernels)
        u, v = data[0], data[1]
        result = euclidean_metric(u, v, get_x_train=lambda: x_train)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result,
                               np.linalg.norm(aks_kernels[0].kernel.K(x_train, x_train) -
                                              aks_kernels[1].kernel.K(x_train, x_train)))

    def test_euclidean_kernel_kernel(self):
        x_train = np.array([[1, 2], [3, 4]])
        result = euclidean_kernel_kernel(x_train)
        self.assertIsInstance(result, KernelKernel)
        self.assertEqual(result.dist_metric, euclidean_metric)


class TestAKSKernel(unittest.TestCase):

    def test_to_binary_tree(self):
        kernel = RBF(1) * RBF(1) + RatQuad(1)
        aks_kernel = AKSKernel(kernel)
        result = aks_kernel.to_binary_tree()
        self.assertIsInstance(result, KernelTree)
        self.assertCountEqual(result.postfix_tokens(), ['SE_0', 'SE_0', '*', 'RQ_0', '+'])


class TestKernelNode(unittest.TestCase):

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


class TestKernelTree(unittest.TestCase):

    def test_init(self):
        kern = RBF(1)
        root = KernelNode(kern)
        result = KernelTree(root)
        self.assertEqual(result.root, root)
        self.assertEqual(result.root.label, 'SE_0')
        self.assertEqual(result.root.value, kern)


if __name__ == '__main__':
    unittest.main()

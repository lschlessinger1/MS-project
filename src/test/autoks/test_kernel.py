import unittest
from unittest.mock import MagicMock

import numpy as np
from GPy.core.parameterization.priors import LogGaussian
from GPy.kern import Add, Prod, RBF, RationalQuadratic, RBFKernelKernel

from src.autoks.core.gp_model import encode_gp_models, GPModel
from src.autoks.distance.metrics import shd_metric, euclidean_metric
from src.autoks.kernel import sort_kernel, get_all_1d_kernels, create_1d_kernel, \
    set_priors, KernelTree, KernelNode, subkernel_expression, \
    decode_kernel, hd_kern_nodes, encode_kernel, additive_part_to_vec, \
    kernel_vec_avg_dist, all_pairs_avg_dist, \
    kernels_to_kernel_vecs, get_priors, tokens_to_kernel_symbols
from src.autoks.kernel_kernel import shd_kernel_kernel, euclidean_kernel_kernel
from src.autoks.symbolic.kernel_symbol import KernelSymbol
from src.autoks.util import remove_duplicates
from src.test.autoks.support.util import has_combo_kernel_type


class TestKernel(unittest.TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RationalQuadratic(1, active_dims=[0])
        self.rq1 = RationalQuadratic(1, active_dims=[1])

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
        kernel_types = [(RationalQuadratic, 0), (RationalQuadratic, 1), (RBF, 0), (RBF, 1)]
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

        kernel_types_inner_1 = [(RationalQuadratic, 1), (RBF, 0), (RBF, 1)]
        prod_1 = result.parts[0]
        for (k_class, dim), part in zip(kernel_types_inner_1, prod_1.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_2 = [(RationalQuadratic, 0), (RationalQuadratic, 1)]
        prod_2 = result.parts[1]
        for (k_class, dim), part in zip(kernel_types_inner_2, prod_2.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_3 = [(RationalQuadratic, 0), (RationalQuadratic, 1), (RBF, 1)]
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
        self.assertIsInstance(result, RationalQuadratic)
        self.assertEqual(result.active_dims[0], 1)

    def test_get_priors(self):
        # Test assertRaises ValueError if not all parameters have priors set
        c = RBF(1)
        self.assertRaises(ValueError, get_priors, c)

        c = RBF(1) * RationalQuadratic(1)
        p1 = LogGaussian(2, 2.21)
        p2 = LogGaussian(1, 2.1)
        p3 = LogGaussian(1, 2)
        c.parameters[0].variance.set_prior(p1, warning=False)
        c.parameters[1].lengthscale.set_prior(p2, warning=False)
        c.parameters[0].lengthscale.set_prior(p3, warning=False)
        c.parameters[1].power.set_prior(p2, warning=False)
        c.parameters[1].variance.set_prior(p1, warning=False)
        result = get_priors(c)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5,))
        np.testing.assert_array_equal(result, [p1, p3, p1, p2, p2])

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

    def test_tokens_to_kernel_symbols(self):
        k1 = RBF(1)
        k2 = RationalQuadratic(1)
        kernel_tokens = [k1, '+', k2]
        actual = tokens_to_kernel_symbols(kernel_tokens)
        expected = [KernelSymbol('SE_0', k1), '+', KernelSymbol('RQ_0', k2)]
        self.assertListEqual(expected, actual)

    def test_additive_part_to_vec(self):
        base_kernels = ['SE', 'RQ']
        n_dims = 2
        k = RBF(1) + RBF(1)
        self.assertRaises(TypeError, additive_part_to_vec, k, base_kernels, n_dims)

        k = RBF(1)
        result = additive_part_to_vec(k, base_kernels=base_kernels, n_dims=2)
        np.testing.assert_array_equal(result, np.array([1, 0, 0, 0]))

        k = RBF(1) * RBF(1) * RBF(1, active_dims=[1]) * RationalQuadratic(1, active_dims=[1])
        result = additive_part_to_vec(k, base_kernels=base_kernels, n_dims=2)
        np.testing.assert_array_equal(result, np.array([2, 1, 0, 1]))

        k = RBF(1) * (RBF(1) + RBF(1))
        self.assertRaises(TypeError, additive_part_to_vec, k, base_kernels, n_dims)

    def test_kernel_vec_avg_dist(self):
        kv1 = [np.array([1, 0, 0])]
        kv2 = [np.array([1, 0, 0])]
        result = kernel_vec_avg_dist(kv1, kv2)
        self.assertIsInstance(result, float)
        self.assertEqual(0, result)

        kv1 = [np.array([1, 0, 0])]
        kv2 = [np.array([1, 0, 4])]
        result = kernel_vec_avg_dist(kv1, kv2)
        self.assertIsInstance(result, float)
        self.assertEqual(4, result)

        kv1 = [np.array([1, 0, 4]), np.array([1, 3, 4])]
        kv2 = [np.array([1, 0, 0]), np.array([1, 0, 0])]
        result = kernel_vec_avg_dist(kv1, kv2)
        self.assertIsInstance(result, float)
        self.assertEqual(9 / 2, result)

    def test_kernels_to_kernel_vecs(self):
        base_kernels = ['SE', 'RQ']
        n_dims = 2
        kerns = [RBF(1), RBF(1) * RBF(1) + RBF(1, active_dims=[1])]
        result = kernels_to_kernel_vecs(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[0], np.array([[1, 0, 0, 0]]))
        np.testing.assert_array_equal(result[1], np.array([[2, 0, 0, 0],
                                                           [0, 1, 0, 0]]))

    def test_all_pairs_avg_dist(self):
        base_kernels = ['SE', 'RQ']
        n_dims = 2
        kerns = [RBF(1), RBF(1)]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        self.assertEqual(0, result)

        kerns = [RBF(1), RBF(1) + RBF(1) * RBF(1, active_dims=[1])]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        # [[1 0 0 0]], [[1 0 0 0],
        #               [1 1 0 0]]
        self.assertEqual(0.5, result)

        kerns = [RBF(1), RBF(1) + RBF(1) * RBF(1, active_dims=[1]), RationalQuadratic(1) * RBF(1) *
                 RationalQuadratic(1) + RBF(1, active_dims=[1])]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        # [[1 0 0 0]],  [[1 0 0 0],
        #               [1 1 0 0]],
        # [[1 0 2 0],
        #  [0 1 0 0]]
        expected = (0.5 + (np.sqrt(2) + 2) / 2 + (2 + np.sqrt(2) + 1 + np.sqrt(5)) / 4) / 3
        self.assertAlmostEqual(expected, result)

    def test_shd_metric(self):
        gp_models = [GPModel(RBF(1) + RationalQuadratic(1)), GPModel(RBF(1))]
        data = encode_gp_models(gp_models)
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

    def test_encode_kernel(self):
        kern = RBF(1, active_dims=[0])
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

        kern = RBF(1, active_dims=[0]) + RBF(1)
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

    def test_shd_kernel_kernel(self):
        result = shd_kernel_kernel(variance=1, lengthscale=1)
        self.assertIsInstance(result, RBFKernelKernel)
        self.assertEqual(result.dist_metric, shd_metric)

        gp_models_evaluated = [GPModel(RBF(1)), GPModel(RationalQuadratic(1))]
        x = encode_gp_models(gp_models_evaluated)
        d = result._unscaled_dist(x)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue(np.array_equal(d, np.array([[0, 1], [1, 0]])))
        k = RBF(1, variance=1, lengthscale=1)
        self.assertTrue(np.array_equal(result.K(x), k.K(d)))

    def test_euclidean_metric(self):
        x_train = np.array([[1, 2], [3, 4]])
        gp_models = [GPModel(RBF(1) + RationalQuadratic(1)), GPModel(RBF(1))]
        data = encode_gp_models(gp_models)
        u, v = data[0], data[1]
        result = euclidean_metric(u, v, get_x_train=lambda: x_train)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result,
                               np.linalg.norm(gp_models[0].kernel.K(x_train, x_train) -
                                              gp_models[1].kernel.K(x_train, x_train)))

    def test_euclidean_kernel_kernel(self):
        x_train = np.array([[1, 2], [3, 4]])
        result = euclidean_kernel_kernel(x_train)
        self.assertIsInstance(result, RBFKernelKernel)
        self.assertEqual(result.dist_metric, euclidean_metric)





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

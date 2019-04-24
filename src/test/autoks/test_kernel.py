import unittest

import numpy as np
from GPy.kern import RBF, RationalQuadratic, RBFKernelKernel

from src.autoks.core.gp_model import encode_gp_models, GPModel
from src.autoks.distance.metrics import shd_metric, euclidean_metric
from src.autoks.kernel import kernel_vec_avg_dist, all_pairs_avg_dist, kernels_to_kernel_vecs, tokens_to_kernel_symbols
from src.autoks.kernel_kernel import shd_kernel_kernel, euclidean_kernel_kernel
from src.autoks.symbolic.kernel_symbol import KernelSymbol
from src.autoks.util import remove_duplicates


class TestKernel(unittest.TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RationalQuadratic(1, active_dims=[0])
        self.rq1 = RationalQuadratic(1, active_dims=[1])

    def test_remove_duplicates(self):
        # simple example
        data = [1, 1, 1, 2, 3, 4, 'a', 'b', True]
        values = [10, 9, 8, '7', '6', False, 4, 3, 2]
        result = remove_duplicates(data, values)
        self.assertEqual(result, [10, '7', '6', False, 4, 3])

        with self.assertRaises(ValueError):
            remove_duplicates([1, 2, 3], ['1', 2])

    def test_tokens_to_kernel_symbols(self):
        k1 = RBF(1)
        k2 = RationalQuadratic(1)
        kernel_tokens = [k1, '+', k2]
        actual = tokens_to_kernel_symbols(kernel_tokens)
        expected = [KernelSymbol('SE_0', k1), '+', KernelSymbol('RQ_0', k2)]
        self.assertListEqual(expected, actual)

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


if __name__ == '__main__':
    unittest.main()

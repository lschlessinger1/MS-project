from unittest import TestCase

import numpy as np
from GPy.kern import RationalQuadratic, RBF, RBFKernelKernel
from graphviz import Source

from src.autoks.backend.kernel import get_all_1d_kernels, RawKernelType
from src.autoks.core.covariance import Covariance, tokens_to_kernel_symbols, kernel_vec_avg_dist, all_pairs_avg_dist, \
    remove_duplicate_kernels
from src.autoks.core.gp_model import GPModel, encode_gp_models
from src.autoks.core.kernel_encoding import KernelTree
from src.autoks.core.kernel_kernel import shd_kernel_kernel, euclidean_kernel_kernel
from src.autoks.distance.metrics import shd_metric, euclidean_metric
from src.autoks.symbolic.kernel_symbol import KernelSymbol


class TestCovariance(TestCase):

    def setUp(self) -> None:
        base_kernels = get_all_1d_kernels(['SE', 'RQ'], 2)
        self.se_0 = base_kernels[0]
        self.se_1 = base_kernels[1]
        self.rq_0 = base_kernels[2]
        self.rq_1 = base_kernels[3]

    def test_create_empty(self):
        self.assertRaises(TypeError, Covariance)

    def test_create_one_d(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertEqual(kern, cov.raw_kernel)
        self.assertListEqual([kern], cov.infix_tokens)
        self.assertIsInstance(cov.infix, str)
        self.assertIsInstance(cov.infix_full, str)
        self.assertEqual('SE_0', cov.infix)
        self.assertEqual('SE_0', cov.postfix)
        self.assertListEqual([kern], cov.postfix_tokens)
        self.assertIsInstance(cov.symbolic_expr, KernelSymbol)
        self.assertEqual(kern, cov.symbolic_expr.kernel_one_d)
        self.assertIsInstance(cov.symbolic_expr_expanded, KernelSymbol)
        self.assertEqual(kern, cov.symbolic_expr_expanded.kernel_one_d)

    def test_to_binary_tree(self):
        kern = self.se_0
        cov = Covariance(kern)
        tree = cov.to_binary_tree()
        self.assertIsInstance(tree, KernelTree)

    def test_canonical(self):
        kern = self.se_0
        cov = Covariance(kern)
        kern = cov.canonical()
        self.assertIsInstance(kern, RawKernelType)

    def test_to_additive_form(self):
        kern = self.se_0
        cov = Covariance(kern)
        kern = cov.to_additive_form()
        self.assertIsInstance(kern, RawKernelType)

    def test_symbolically_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

    def test_symbolic_expanded_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolic_expanded_equals(cov_2))
        self.assertFalse(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolic_expanded_equals(cov_2))
        self.assertFalse(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        # Test additive equivalence
        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

    def test_infix_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.infix_equals(cov_2))
        self.assertTrue(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_0 * self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.infix_equals(cov_2))
        self.assertTrue(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        # Test additive equivalence
        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

    def test_is_base(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertTrue(cov.is_base())

        kern = self.se_0 + self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_base())

    def test_is_sum(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_sum())

        kern = self.se_0 + self.se_0
        cov = Covariance(kern)
        self.assertTrue(cov.is_sum())

        kern = self.se_0 * self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_sum())

    def test_is_prod(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_prod())

        kern = self.se_0 + self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_prod())

        kern = self.se_0 * self.se_0
        cov = Covariance(kern)
        self.assertTrue(cov.is_prod())

    def test_add(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)

        cov = cov_1 + cov_2
        self.assertEqual('SE_0 + SE_0', cov.infix)

    def test_multiply(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)

        cov = cov_1 * cov_2
        self.assertEqual('SE_0 * SE_0', cov.infix)

    def test_as_latex(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_latex()
        self.assertIsInstance(actual, str)

    def test_as_mathml(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_mathml()
        self.assertIsInstance(actual, str)

    def test_as_dot(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_dot()
        self.assertIsInstance(actual, str)

    def test_as_graph(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_graph()
        self.assertIsInstance(actual, Source)


class TestCovarianceModule(TestCase):

    def setUp(self):
        self.se0 = Covariance(RBF(1, active_dims=[0]))
        self.se1 = Covariance(RBF(1, active_dims=[1]))
        self.rq0 = Covariance(RationalQuadratic(1, active_dims=[0]))
        self.rq1 = Covariance(RationalQuadratic(1, active_dims=[1]))

    def test_remove_duplicate_kernels(self):
        covariances = [self.se0 + self.se0,
                       self.se1,
                       self.se0,
                       self.se0,
                       self.se1 + self.se0,
                       self.se0 + self.se1]
        kernels_pruned = remove_duplicate_kernels(covariances)
        # should be [SE_0 + SE_0, SE_1, SE_0, SE1 + SE0]
        expected_kernels = covariances[:3] + [covariances[4]]
        new_kernels_infixes = [k.infix for k in kernels_pruned]
        expected_infixes = [k.infix for k in expected_kernels]
        self.assertListEqual(expected_infixes, new_kernels_infixes)

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

    def test_all_pairs_avg_dist(self):
        base_kernels = ['SE', 'RQ']
        n_dims = 2
        kerns = [self.se0, self.se0]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        self.assertEqual(0, result)

        kerns = [self.se0, self.se0 + self.se0 * self.se1]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        # [[1 0 0 0]], [[1 0 0 0],
        #               [1 1 0 0]]
        self.assertEqual(0.5, result)

        kerns = [self.se0, self.se0 + self.se0 * self.se1, self.rq0 * self.se0 * self.rq0 + self.se1]
        result = all_pairs_avg_dist(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, float)
        # [[1 0 0 0]],  [[1 0 0 0],
        #               [1 1 0 0]],
        # [[1 0 2 0],
        #  [0 1 0 0]]
        expected = (0.5 + (np.sqrt(2) + 2) / 2 + (2 + np.sqrt(2) + 1 + np.sqrt(5)) / 4) / 3
        self.assertAlmostEqual(expected, result)

    def test_shd_metric(self):
        gp_models = [GPModel(Covariance(RBF(1) + RationalQuadratic(1))), GPModel(Covariance(RBF(1)))]
        data = encode_gp_models(gp_models)
        u, v = data[0], data[1]
        result = shd_metric(u, v)
        self.assertEqual(result, 1)

    def test_shd_kernel_kernel(self):
        result = shd_kernel_kernel(variance=1, lengthscale=1)
        self.assertIsInstance(result, RBFKernelKernel)
        self.assertEqual(result.dist_metric, shd_metric)

        gp_models_evaluated = [GPModel(Covariance(RBF(1))), GPModel(Covariance(RationalQuadratic(1)))]
        x = encode_gp_models(gp_models_evaluated)
        d = result._unscaled_dist(x)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue(np.array_equal(d, np.array([[0, 1], [1, 0]])))
        k = RBF(1, variance=1, lengthscale=1)
        self.assertTrue(np.array_equal(result.K(x), k.K(d)))

    def test_euclidean_metric(self):
        x_train = np.array([[1, 2], [3, 4]])
        gp_models = [GPModel(Covariance(RBF(1) + RationalQuadratic(1))), GPModel(Covariance(RBF(1)))]
        data = encode_gp_models(gp_models)
        u, v = data[0], data[1]
        result = euclidean_metric(u, v, get_x_train=lambda: x_train)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result,
                               np.linalg.norm(gp_models[0].covariance.raw_kernel.K(x_train, x_train) -
                                              gp_models[1].covariance.raw_kernel.K(x_train, x_train)))

    def test_euclidean_kernel_kernel(self):
        x_train = np.array([[1, 2], [3, 4]])
        result = euclidean_kernel_kernel(x_train)
        self.assertIsInstance(result, RBFKernelKernel)
        self.assertEqual(result.dist_metric, euclidean_metric)

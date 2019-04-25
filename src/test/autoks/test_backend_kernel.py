from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.core.parameterization.priors import LogGaussian
from GPy.kern import RBF, RationalQuadratic, Add, Prod

from src.autoks.backend.kernel import get_kernel_mapping, get_allowable_kernels, get_matching_kernels, create_1d_kernel, \
    get_priors, set_priors, subkernel_expression, sort_kernel, get_all_1d_kernels, additive_part_to_vec, decode_kernel, \
    encode_kernel, kernels_to_kernel_vecs
from src.test.autoks.support.util import has_combo_kernel_type


class TestBackendKernel(TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RationalQuadratic(1, active_dims=[0])
        self.rq1 = RationalQuadratic(1, active_dims=[1])

    def test_get_kernel_mapping(self):
        actual = get_kernel_mapping()
        self.assertIsInstance(actual, dict)
        self.assertIsInstance(list(actual.keys())[0], str)

    def test_get_allowable_kernels(self):
        actual = get_allowable_kernels()
        self.assertIsInstance(actual, list)
        self.assertIsInstance(actual[0], str)

    def test_get_matching_kernels(self):
        actual = get_matching_kernels()
        self.assertIsInstance(actual, list)

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

    def test_kernels_to_kernel_vecs(self):
        base_kernels = ['SE', 'RQ']
        n_dims = 2
        kerns = [self.se0, self.se0 * self.se0 + self.se1]
        result = kernels_to_kernel_vecs(kerns, base_kernels, n_dims)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[0], np.array([[1, 0, 0, 0]]))
        np.testing.assert_array_equal(result[1], np.array([[2, 0, 0, 0],
                                                           [0, 1, 0, 0]]))

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

    def test_decode_kernel(self):
        kern = RBF(1)
        kern_dict = kern.to_dict()
        kern_dict_str = str(kern_dict)
        result = decode_kernel(kern_dict_str)
        self.assertIsInstance(result, RBF)
        self.assertEqual(result.input_dim, 1)
        self.assertDictEqual(result.to_dict(), kern_dict)

    def test_encode_kernel(self):
        kern = RBF(1, active_dims=[0])
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

        kern = RBF(1, active_dims=[0]) + RBF(1)
        result = encode_kernel(kern)
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(kern.to_dict()))

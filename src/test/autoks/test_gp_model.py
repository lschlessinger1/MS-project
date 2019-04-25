from unittest import TestCase

import numpy as np
from GPy.kern import RationalQuadratic, RBF

from src.autoks.backend.kernel import encode_kernel
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import remove_duplicate_gp_models, GPModel, encode_gp_model, encode_gp_models
from src.autoks.core.kernel_encoding import KernelTree


class TestGPModel(TestCase):

    def test_to_binary_tree(self):
        kernel = Covariance(RBF(1) * RBF(1) + RationalQuadratic(1))
        gp_model = GPModel(kernel)
        result = gp_model.covariance.to_binary_tree()
        self.assertIsInstance(result, KernelTree)
        self.assertCountEqual(result.postfix_tokens(), ['SE_0', 'SE_0', '*', 'RQ_0', '+'])


class TestGPModelModule(TestCase):

    def test_remove_duplicate_gp_models(self):
        k1 = GPModel(Covariance(RBF(1)))
        k1.score = 10

        k2 = GPModel(Covariance(RBF(1)))
        k2.score = 9

        k3 = GPModel(Covariance(RBF(1)))
        k3.nan_scored = True

        k4 = GPModel(Covariance(RBF(1)))
        k4.nan_scored = True

        k5 = GPModel(Covariance(RBF(1)))

        k6 = GPModel(Covariance(RBF(1, lengthscale=0.5)))

        k7 = GPModel(Covariance(RationalQuadratic(1)))

        # Always keep k1 then k2 then k3 etc.
        result = remove_duplicate_gp_models([k1, k2, k3, k4, k5, k6, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k1, k2, k3, k4, k5, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k1, k2, k3, k4, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k1, k2, k3, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k1, k2, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k1, k7])
        self.assertListEqual(result, [k1, k7])

        result = remove_duplicate_gp_models([k2, k3, k4, k5, k6, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_gp_models([k2, k3, k4, k5, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_gp_models([k2, k3, k4, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_gp_models([k2, k3, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_gp_models([k2, k7])
        self.assertListEqual(result, [k2, k7])

        result = remove_duplicate_gp_models([k3, k4, k5, k6, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k3, k4, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k4, k3, k5, k6, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k4, k3, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k3, k4, k5, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k3, k4, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k4, k3, k7])
        self.assertTrue(result == [k3, k7] or result == [k4, k7])

        result = remove_duplicate_gp_models([k3, k7])
        self.assertListEqual(result, [k3, k7])

        result = remove_duplicate_gp_models([k4, k7])
        self.assertListEqual(result, [k4, k7])

        result = remove_duplicate_gp_models([k5, k6, k7])
        self.assertTrue(result == [k5, k7] or result == [k6, k7])

        result = remove_duplicate_gp_models([k6, k5, k7])
        self.assertTrue(result == [k5, k7] or result == [k6, k7])

    def test_encode_gp_model(self):
        gp_model = GPModel(Covariance(RBF(1, active_dims=[0])))
        result = encode_gp_model(gp_model)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], str)
        self.assertListEqual(result, [encode_kernel(gp_model.covariance.raw_kernel), [None]])

    def test_encode_gp_models(self):
        gp_models = [GPModel(Covariance(RBF(1))), GPModel(Covariance(RationalQuadratic(1)))]
        result = encode_gp_models(gp_models)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(gp_models), 1))
        self.assertListEqual(result[0][0], [encode_kernel(gp_models[0].covariance.raw_kernel), [None]])
        self.assertListEqual(result[1][0], [encode_kernel(gp_models[1].covariance.raw_kernel), [None]])

        gp_models = [GPModel(Covariance(RBF(1) * RBF(1))), GPModel(Covariance(RationalQuadratic(1)))]
        result = encode_gp_models(gp_models)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (len(gp_models), 1))
        self.assertListEqual(result[0][0], [encode_kernel(gp_models[0].covariance.raw_kernel), [None]])
        self.assertListEqual(result[1][0], [encode_kernel(gp_models[1].covariance.raw_kernel), [None]])

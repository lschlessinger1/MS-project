from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.core.parameterization.priors import LogGaussian, Gaussian
from GPy.kern import RBF, RationalQuadratic
from numpy.linalg import LinAlgError

from src.autoks.distance.distance import fix_numerical_problem, chol_safe, HellingerDistanceBuilder


class TestDistance(TestCase):

    def test_fix_numerical_problem(self):
        tolerance = 0.1

        k = np.array([[1, 2],
                      [3, 4]])
        result = fix_numerical_problem(k, tolerance)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, k.shape)

        # Test singular matrix does not raise LinAlgError
        k = np.ones((3, 3))
        result = fix_numerical_problem(k, tolerance)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, k.shape)

        # Test non-square matrix assertRaises LinAlgError
        k = np.ones((3, 1))
        self.assertRaises(LinAlgError, fix_numerical_problem, k, tolerance)

    def test_chol_safe(self):
        tolerance = 1e-6

        # Test non-singular matrix
        k = np.array([[1, 2],
                      [3, 4]])
        result = chol_safe(k, tolerance)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, k.shape)

        # Test singular matrix (fail IF raises LinAlgError)
        k = np.ones((3, 3))
        result = chol_safe(k, tolerance)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, k.shape)

        # Test non-square matrix assertRaises LinAlgError
        k = np.ones((3, 1))
        self.assertRaises(LinAlgError, chol_safe, k, tolerance)


class TestHellingerDistanceBuilder(TestCase):

    def test_hellinger_distance(self):
        cov_i = RBF(1)
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        cov_i.variance.set_prior(p1, warning=False)
        cov_i.lengthscale.set_prior(p2, warning=False)

        cov_j = RationalQuadratic(1)
        p3 = LogGaussian(11, 2)
        p4 = LogGaussian(2, 1.12)
        cov_j.variance.set_prior(p3, warning=False)
        cov_j.lengthscale.set_prior(p4, warning=False)
        cov_j.power.set_prior(p2, warning=False)

        x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        builder = HellingerDistanceBuilder(noise_prior, num_samples=20, max_num_hyperparameters=40, max_num_kernels=10,
                                           active_models=MagicMock(), initial_model_indices=MagicMock(), data_X=x)
        log_det_i, mini_gram_matrices_i = builder.create_precomputed_info(cov_i, x)
        log_det_j, mini_gram_matrices_j = builder.create_precomputed_info(cov_j, x)
        result = HellingerDistanceBuilder.hellinger_distance(log_det_i, mini_gram_matrices_i,
                                                             log_det_j, mini_gram_matrices_j)
        self.assertIsInstance(result, float)

    def test_create_precomputed_info(self):
        num_samples = 20
        x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        builder = HellingerDistanceBuilder(noise_prior, num_samples, max_num_hyperparameters=40, max_num_kernels=10,
                                           active_models=MagicMock(), initial_model_indices=MagicMock(), data_X=x)

        cov = RBF(1)
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        cov.variance.set_prior(p1, warning=False)
        cov.lengthscale.set_prior(p2, warning=False)

        result = builder.create_precomputed_info(cov, x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertEqual(result[0].shape, (num_samples,))
        self.assertEqual(result[1].shape, (x.shape[0], x.shape[0], num_samples))

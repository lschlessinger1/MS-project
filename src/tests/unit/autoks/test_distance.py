from typing import List
from unittest import TestCase, mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from GPy.core.parameterization.priors import LogGaussian, Gaussian
from GPy.kern import RBF, RationalQuadratic
from numpy.linalg import LinAlgError

from src.autoks.core.active_set import ActiveSet
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import BomsGrammar
from src.autoks.distance.distance import fix_numerical_problem, chol_safe, HellingerDistanceBuilder, DistanceBuilder, \
    FrobeniusDistanceBuilder, CorrelationDistanceBuilder


class DistanceMetricCase:

    def __init__(self, metric, pool, n):
        self.metric = metric
        self.pool = pool
        self.n = n

    @pytest.mark.slow
    def test_non_negativity(self):
        """Test distance metric non-negativity for n samples.

        d(x, y) >= 0
        """
        # pool = np.asarray(self.pool)
        for _ in range(self.n):
            ind = np.random.choice(len(self.pool), size=2)
            x, y = (self.pool[i] for i in ind)
            distance = self.metric(x, y)
            assert distance >= 0

    @pytest.mark.slow
    def test_identity_of_indiscernibles(self):
        """Test distance metric identity of indiscernibles for n samples.

        Also known as Leibniz's law

        d(x, y) = 0 <=> x = y
        """
        for _ in range(self.n):
            ind = np.random.choice(len(self.pool), size=1)[0]
            x = self.pool[ind]
            distance = self.metric(x, x)
            expected = 0
            assert distance == pytest.approx(expected, abs=1e-3)

    @pytest.mark.slow
    def test_symmetry(self):
        """Test distance metric symmetry for n samples.

        d(x, y) = d(y, x)
        """

        for _ in range(self.n):
            ind = np.random.choice(len(self.pool), size=2)
            x, y = (self.pool[i] for i in ind)

            distance_1 = self.metric(x, y)
            distance_2 = self.metric(y, x)
            # doesn't work if x == y
            diff = distance_1 - distance_2
            expected = 0.
            assert diff == pytest.approx(expected, abs=1e-2)

    @pytest.mark.slow
    def test_triangle_inequality(self):
        """Test distance metric triangle inequality for n samples.

        d(x, z) <= d(x, y) + d(y, z)
        """
        for _ in range(self.n):
            ind = np.random.choice(len(self.pool), size=3)
            x, y, z = (self.pool[i] for i in ind)

            tol = 1e-3
            assert self.metric(x, z) - tol <= self.metric(x, y) + self.metric(y, z)


class BoundedMetricCase(DistanceMetricCase):

    @pytest.mark.slow
    def test_boundedness(self):
        """Test distance metric boundedness for n samples.

        d(x, y) <= b
        """
        for _ in range(self.n):
            ind = np.random.choice(len(self.pool), size=2)
            x, y = (self.pool[i] for i in ind)
            distance = self.metric(x, y)
            assert distance <= self.bound


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


class TestDistanceBuilder(TestCase):

    def setUp(self) -> None:
        self.x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        self.noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        self.cov_i = RBF(1)
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        self.cov_i.variance.set_prior(p1, warning=False)
        self.cov_i.lengthscale.set_prior(p2, warning=False)
        models = [GPModel(self.cov_i)]
        self.active_models = ActiveSet(max_n_models=3)
        self.active_models.models = models
        self.ind_init = [0]

    @mock.patch('src.autoks.distance.distance.DistanceBuilder.precompute_information')
    def test_init(self, mock_precompute_information):
        builder = DistanceBuilder(self.noise_prior, num_samples=20, max_num_hyperparameters=40, max_num_kernels=3,
                                  active_models=self.active_models, initial_model_indices=self.ind_init, data_X=self.x)
        ps = builder.probability_samples
        self.assertIsInstance(ps, np.ndarray)
        self.assertEqual(ps.shape, (builder.num_samples, builder.max_num_hyperparameters))

        hdns = builder.hyperparameter_data_noise_samples
        self.assertIsInstance(hdns, np.ndarray)
        self.assertEqual(hdns.shape, (builder.num_samples, 1))

        ad = builder._average_distance
        self.assertIsInstance(ad, np.ndarray)
        self.assertEqual(ad.shape, (builder.max_num_kernels, builder.max_num_kernels))
        nan = np.nan
        np.testing.assert_equal(ad, np.array([[0, nan, nan],
                                              [nan, 0, nan],
                                              [nan, nan, 0]]))

        mock_precompute_information.assert_called_once_with(self.active_models, self.ind_init, self.x)

    def test_metric(self):
        self.assertRaises(NotImplementedError, DistanceBuilder.metric, 'arg1', 'arg2')


class TestHellingerDistanceBuilder(TestCase):

    def setUp(self) -> None:
        self.x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])

        self.noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        cov_1 = Covariance(RBF(1))
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        cov_1.raw_kernel.variance.set_prior(p1, warning=False)
        cov_1.raw_kernel.lengthscale.set_prior(p2, warning=False)

        cov_2 = Covariance(RBF(1))
        p3 = LogGaussian(11, 1)
        p4 = LogGaussian(1, 1.21)
        cov_2.raw_kernel.variance.set_prior(p3, warning=False)
        cov_2.raw_kernel.lengthscale.set_prior(p4, warning=False)

        cov_3 = Covariance(RationalQuadratic(1))
        p5 = LogGaussian(4, 1)
        p6 = LogGaussian(1.2, 1.21)
        p7 = LogGaussian(13, 1.21)
        cov_3.raw_kernel.variance.set_prior(p5, warning=False)
        cov_3.raw_kernel.lengthscale.set_prior(p6, warning=False)
        cov_3.raw_kernel.power.set_prior(p7, warning=False)
        models = [GPModel(cov_1), GPModel(cov_2), GPModel(cov_3)]
        self.active_models = ActiveSet(max_n_models=3)
        self.active_models.models = models
        self.ind_init = [0, 2]

    def test_precompute_information(self):
        n = self.x.shape[0]
        builder = HellingerDistanceBuilder(self.noise_prior, num_samples=20, max_num_hyperparameters=40,
                                           max_num_kernels=3, active_models=self.active_models,
                                           initial_model_indices=self.ind_init, data_X=self.x)

        self.assertIsInstance(self.active_models.models[0].info, tuple)
        self.assertIsInstance(self.active_models.models[0].info[0], np.ndarray)
        self.assertIsInstance(self.active_models.models[0].info[1], np.ndarray)
        self.assertEqual(self.active_models.models[0].info[0].shape, (builder.num_samples,))
        self.assertEqual(self.active_models.models[0].info[1].shape, (n, n, builder.num_samples,))

        new_ind = [1]
        builder.precompute_information(self.active_models, new_ind, self.x)
        self.assertIsInstance(self.active_models.models[1].info, tuple)
        self.assertIsInstance(self.active_models.models[1].info[0], np.ndarray)
        self.assertIsInstance(self.active_models.models[1].info[1], np.ndarray)
        self.assertEqual(self.active_models.models[1].info[0].shape, (builder.num_samples,))

        self.assertEqual(self.active_models.models[1].info[1].shape, (n, n, builder.num_samples,))

    def test_update(self):
        builder = HellingerDistanceBuilder(self.noise_prior, num_samples=20, max_num_hyperparameters=40,
                                           max_num_kernels=3, active_models=self.active_models,
                                           initial_model_indices=self.ind_init, data_X=self.x)

        builder.update(self.active_models, [1], [self.ind_init[0]], [self.ind_init[1]], self.x)

    def test_get_kernel(self):
        builder = HellingerDistanceBuilder(self.noise_prior, num_samples=20, max_num_hyperparameters=40,
                                           max_num_kernels=3, active_models=self.active_models,
                                           initial_model_indices=self.ind_init, data_X=self.x)

        result = builder.get_kernel(1)
        self.assertIsInstance(result, np.ndarray)

    def test_compute_distance(self):
        builder = HellingerDistanceBuilder(self.noise_prior, num_samples=20, max_num_hyperparameters=40,
                                           max_num_kernels=3, active_models=self.active_models,
                                           initial_model_indices=self.ind_init, data_X=self.x)

        builder.compute_distance(self.active_models, [self.ind_init[0]], [self.ind_init[1]])
        # check symmetric:
        np.testing.assert_allclose(builder._average_distance, builder._average_distance.T, atol=1e-8)

    def test_hellinger_distance(self):
        cov_i = Covariance(RBF(1))
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        cov_i.raw_kernel.variance.set_prior(p1, warning=False)
        cov_i.raw_kernel.lengthscale.set_prior(p2, warning=False)

        cov_j = Covariance(RationalQuadratic(1))
        p3 = LogGaussian(11, 2)
        p4 = LogGaussian(2, 1.12)
        cov_j.raw_kernel.variance.set_prior(p3, warning=False)
        cov_j.raw_kernel.lengthscale.set_prior(p4, warning=False)
        cov_j.raw_kernel.power.set_prior(p2, warning=False)

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

        cov = Covariance(RBF(1))
        p1 = LogGaussian(20, 1)
        p2 = LogGaussian(0, 1.1)
        cov.raw_kernel.variance.set_prior(p1, warning=False)
        cov.raw_kernel.lengthscale.set_prior(p2, warning=False)

        result = builder.create_precomputed_info(cov, x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertEqual(result[0].shape, (num_samples,))
        self.assertEqual(result[1].shape, (x.shape[0], x.shape[0], num_samples))


class TestHellingerDistanceMetric(TestCase, BoundedMetricCase):

    def setUp(self) -> None:
        x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        builder = HellingerDistanceBuilder(noise_prior, num_samples=20, max_num_hyperparameters=40, max_num_kernels=10,
                                           active_models=MagicMock(), initial_model_indices=MagicMock(), data_X=x)

        self.metric = builder.metric
        self.n = 50
        pool = create_pool_of_covariances(self.n)
        self.pool = [builder.create_precomputed_info(cov, x) for cov in pool]
        self.bound = 1


class TestCorrelationDistanceMetric(TestCase, BoundedMetricCase):

    def setUp(self) -> None:
        x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        builder = CorrelationDistanceBuilder(noise_prior, num_samples=20, max_num_hyperparameters=40,
                                             max_num_kernels=10,
                                             active_models=MagicMock(), initial_model_indices=MagicMock(), data_X=x)

        self.metric = builder.metric
        self.n = 50
        pool = create_pool_of_covariances(self.n)
        self.pool = [builder.create_precomputed_info(cov, x) for cov in pool]
        self.bound = 1


class TestFrobeniusDistanceMetric(TestCase, DistanceMetricCase):

    def setUp(self) -> None:
        x = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]])
        noise_prior = Gaussian(mu=np.log(0.01), sigma=1)
        builder = FrobeniusDistanceBuilder(noise_prior, num_samples=20, max_num_hyperparameters=40, max_num_kernels=10,
                                           active_models=MagicMock(), initial_model_indices=MagicMock(), data_X=x)

        self.metric = builder.metric
        self.n = 50
        pool = create_pool_of_covariances(self.n)
        self.pool = [builder.create_precomputed_info(cov, x) for cov in pool]


def create_pool_of_covariances(n: int = 50, n_dims: int = 1) -> List[Covariance]:
    grammar = BomsGrammar(base_kernel_names=['SE', 'RQ', 'LIN', 'PER'])
    grammar.build(n_dims=n_dims)
    max_number_of_models = n

    level = 3
    covariances = grammar.expand_full_brute_force(level, max_number_of_models)
    return covariances

from unittest import TestCase

import numpy as np
from GPy.core.parameterization.priors import Gaussian, LogGaussian

from src.autoks.distance.util import probability_samples, prior_sample_gaussian, prior_sample, prior_sample_log_gaussian


class TestDistanceUtil(TestCase):

    def test_probability_samples(self):
        m = 15
        n = 10
        result = probability_samples(max_num_hyperparameters=m, num_samples=n)
        assert isinstance(result, np.ndarray)
        assert result.shape == (n, m)

    def test_prior_sample(self):
        prior_1 = Gaussian(mu=0.5, sigma=2)
        prior_2 = LogGaussian(mu=0.5, sigma=2)
        prior_3 = Gaussian(mu=0.7, sigma=2.2)
        prior_4 = LogGaussian(mu=0.7, sigma=2.2)
        prior_list = np.array([prior_1, prior_2, prior_3, prior_4])

        samples = probability_samples()
        result = prior_sample(prior_list, samples)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (samples.shape[0], prior_list.size))

    def test_prior_sample_gaussian_fast(self):
        prior_1 = Gaussian(mu=0.5, sigma=2)
        prior_2 = Gaussian(mu=0.7, sigma=2.2)
        prior_list = np.array([prior_1, prior_2])

        samples = probability_samples()
        result = prior_sample_gaussian(prior_list, samples)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (samples.shape[0], prior_list.size))

    def test_prior_gaussian_sample_slow(self):
        prior_1 = Gaussian(mu=0.5, sigma=2)
        prior_2 = Gaussian(mu=0.7, sigma=2.2)
        prior_list = np.array([prior_1, prior_2])
        samples = probability_samples(num_samples=2000)
        prior_samp = prior_sample_gaussian(prior_list, samples)
        # Test mean and standard deviation are within 0.1 error from expected
        np.testing.assert_approx_equal(prior_samp[:, 0].std(), prior_list[0].sigma, significant=1)
        np.testing.assert_approx_equal(prior_samp[:, 0].mean(), prior_list[0].mu, significant=1)
        np.testing.assert_approx_equal(prior_samp[:, 1].std(), prior_list[1].sigma, significant=1)
        np.testing.assert_approx_equal(prior_samp[:, 1].mean(), prior_list[1].mu, significant=1)

    def test_prior_log_gaussian_fast(self):
        prior_1 = LogGaussian(mu=0.5, sigma=2)
        prior_2 = LogGaussian(mu=0.7, sigma=2.2)
        prior_list = np.array([prior_1, prior_2])

        samples = probability_samples()
        result = prior_sample_log_gaussian(prior_list, samples)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (samples.shape[0], prior_list.size))

from unittest import TestCase

import numpy as np

from src.autoks.core.hyperprior import boms_hyperpriors, HyperpriorMap
from src.autoks.core.prior import PriorDist
from src.evalg.serialization import Serializable


class TestHyperprior(TestCase):

    def test_boms_hyperpriors(self):
        result = boms_hyperpriors()

        self.assertIsInstance(result, HyperpriorMap)

        self.assertIn('SE', result)
        self.assertIn('RQ', result)
        self.assertIn('LIN', result)
        self.assertIn('PER', result)
        self.assertIn('GP', result)

        self.assertEqual(5, len(result))

        self.assertIsInstance(result['SE']['variance'], PriorDist)
        self.assertIsInstance(result['SE']['lengthscale'], PriorDist)

        self.assertIsInstance(result['RQ']['variance'], PriorDist)
        self.assertIsInstance(result['RQ']['lengthscale'], PriorDist)
        self.assertIsInstance(result['RQ']['power'], PriorDist)

        self.assertIsInstance(result['PER']['variance'], PriorDist)
        self.assertIsInstance(result['PER']['lengthscale'], PriorDist)
        self.assertIsInstance(result['PER']['period'], PriorDist)

        self.assertIsInstance(result['LIN']['variances'], PriorDist)
        self.assertIsInstance(result['LIN']['shifts'], PriorDist)

        self.assertIsInstance(result['GP']['variance'], PriorDist)

        self.assertEqual(np.log(0.4), result['SE']['variance'].raw_prior.mu)
        self.assertEqual(1, result['SE']['variance'].raw_prior.sigma)
        self.assertEqual(np.log(0.1), result['SE']['lengthscale'].raw_prior.mu)
        self.assertEqual(1, result['SE']['lengthscale'].raw_prior.sigma)

        self.assertEqual(np.log(0.4), result['RQ']['variance'].raw_prior.mu)
        self.assertEqual(1, result['RQ']['variance'].raw_prior.sigma)
        self.assertEqual(np.log(0.1), result['RQ']['lengthscale'].raw_prior.mu)
        self.assertEqual(1, result['RQ']['lengthscale'].raw_prior.sigma)
        self.assertEqual(np.log(0.05), result['RQ']['power'].raw_prior.mu)
        self.assertEqual(np.sqrt(0.5), result['RQ']['power'].raw_prior.sigma)

        self.assertEqual(np.log(0.4), result['PER']['variance'].raw_prior.mu)
        self.assertEqual(1, result['PER']['variance'].raw_prior.sigma)
        self.assertEqual(np.log(2), result['PER']['lengthscale'].raw_prior.mu)
        self.assertEqual(np.sqrt(0.5), result['PER']['lengthscale'].raw_prior.sigma)
        self.assertEqual(np.log(0.1), result['PER']['period'].raw_prior.mu)
        self.assertEqual(np.sqrt(0.5), result['PER']['period'].raw_prior.sigma)

        self.assertEqual(np.log(0.4), result['LIN']['variances'].raw_prior.mu)
        self.assertEqual(1, result['LIN']['variances'].raw_prior.sigma)
        self.assertEqual(0, result['LIN']['shifts'].raw_prior.mu)
        self.assertEqual(1, result['LIN']['shifts'].raw_prior.sigma)

        self.assertEqual(np.log(0.01), result['GP']['variance'].raw_prior.mu)
        self.assertEqual(np.sqrt(0.1), result['GP']['variance'].raw_prior.sigma)

    def test_to_dict(self):
        prior_map = dict()
        prior_l = PriorDist.from_prior_str("LOGNORMAL", {'mu': 0.1, 'sigma': 0.7 ** 2})
        prior_sigma = PriorDist.from_prior_str("LOGNORMAL", {'mu': 0.4, 'sigma': 0.7 ** 2})
        prior_map['SE'] = {
            'variance': prior_sigma,
            'lengthscale': prior_l
        }
        hyperprior_map = HyperpriorMap(prior_map)

        actual = hyperprior_map.to_dict()

        self.assertIsInstance(actual, dict)
        self.assertIn('prior_map', actual)
        self.assertIn('SE', actual['prior_map'])
        self.assertIn('variance', actual['prior_map']['SE'])
        self.assertIn('lengthscale', actual['prior_map']['SE'])
        self.assertEqual(prior_map['SE']['variance'].to_dict(), actual['prior_map']['SE']['variance'])
        self.assertEqual(prior_map['SE']['lengthscale'].to_dict(), actual['prior_map']['SE']['lengthscale'])

    def test_from_dict(self):
        test_cases = (HyperpriorMap, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                prior_map = dict()
                prior_l = PriorDist.from_prior_str("LOGNORMAL", {'mu': 0.1, 'sigma': 0.7 ** 2})
                prior_sigma = PriorDist.from_prior_str("LOGNORMAL", {'mu': 0.4, 'sigma': 0.7 ** 2})
                prior_map['SE'] = {
                    'variance': prior_sigma,
                    'lengthscale': prior_l
                }
                hyperprior_map = HyperpriorMap(prior_map)

                actual = cls.from_dict(hyperprior_map.to_dict())

                self.assertIsInstance(actual, HyperpriorMap)
                self.assertIn('SE', actual)
                self.assertIn('variance', actual['SE'])
                self.assertIn('lengthscale', actual['SE'])
                self.assertEqual(prior_map['SE']['variance'].__class__, actual['SE']['variance'].__class__)
                self.assertEqual(prior_map['SE']['lengthscale'].__class__, actual['SE']['lengthscale'].__class__)

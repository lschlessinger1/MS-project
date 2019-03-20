from unittest import TestCase

from GPy.core.parameterization.priors import LogGaussian

from src.autoks.hyperprior import boms_hyperpriors


class TestGrammar(TestCase):

    def test_boms_hyperpriors(self):
        result = boms_hyperpriors()

        self.assertIsInstance(result, dict)

        self.assertIn('SE', result)
        self.assertIn('RQ', result)
        self.assertIn('LIN', result)
        self.assertIn('PER', result)
        self.assertIn('GP', result)

        self.assertEqual(len(result), 5)

        self.assertIsInstance(result['SE']['variance'], LogGaussian)
        self.assertIsInstance(result['SE']['lengthscale'], LogGaussian)

        self.assertIsInstance(result['RQ']['variance'], LogGaussian)
        self.assertIsInstance(result['RQ']['lengthscale'], LogGaussian)
        self.assertIsInstance(result['RQ']['power'], LogGaussian)

        self.assertIsInstance(result['PER']['variance'], LogGaussian)
        self.assertIsInstance(result['PER']['lengthscale'], LogGaussian)
        self.assertIsInstance(result['PER']['period'], LogGaussian)

        self.assertIsInstance(result['LIN']['variances'], LogGaussian)
        self.assertIsInstance(result['LIN']['shifts'], LogGaussian)

        self.assertIsInstance(result['GP']['variance'], LogGaussian)

        self.assertEqual(result['SE']['variance'].mu, 0.4)
        self.assertEqual(result['SE']['variance'].sigma, 0.7 ** 2)
        self.assertEqual(result['SE']['lengthscale'].mu, 0.1)
        self.assertEqual(result['SE']['lengthscale'].sigma, 0.7 ** 2)

        self.assertEqual(result['RQ']['variance'].mu, 0.4)
        self.assertEqual(result['RQ']['variance'].sigma, 0.7 ** 2)
        self.assertEqual(result['RQ']['lengthscale'].mu, 0.1)
        self.assertEqual(result['RQ']['lengthscale'].sigma, 0.7 ** 2)
        self.assertEqual(result['RQ']['power'].mu, 0.05)
        self.assertEqual(result['RQ']['power'].sigma, 0.7 ** 2)

        self.assertEqual(result['PER']['variance'].mu, 0.4)
        self.assertEqual(result['PER']['variance'].sigma, 0.7 ** 2)
        self.assertEqual(result['PER']['lengthscale'].mu, 0.1)
        self.assertEqual(result['PER']['lengthscale'].sigma, 0.7 ** 2)
        self.assertEqual(result['PER']['period'].mu, 2)
        self.assertEqual(result['PER']['period'].sigma, 0.7 ** 2)

        self.assertEqual(result['LIN']['variances'].mu, 0.4)
        self.assertEqual(result['LIN']['variances'].sigma, 0.7 ** 2)
        self.assertEqual(result['LIN']['shifts'].mu, 0)
        self.assertEqual(result['LIN']['shifts'].sigma, 2 ** 2)

        self.assertEqual(result['GP']['variance'].mu, 0.1)
        self.assertEqual(result['GP']['variance'].sigma, 1 ** 2)

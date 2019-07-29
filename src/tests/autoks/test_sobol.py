from unittest import TestCase

import numpy as np

from src.autoks.distance.sobol import gen_sobol, sobol_sample


class TestSobol(TestCase):

    def test_gen_sobol(self):
        result = gen_sobol(n=4, d=2, skip=0)
        np.testing.assert_array_equal(result, np.array([[0., 0.], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]))

        result = gen_sobol(n=3, d=2, skip=1)
        np.testing.assert_array_equal(result, np.array([[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]))

    def test_sobol_sample(self):
        d = 2
        n = 3
        result = sobol_sample(n, d, skip=1000, leap=100)
        # TODO: tests same result as MATLAB `sobolset`
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (n, d))
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

        d = 2
        n = 4
        result = sobol_sample(n, d, skip=0, leap=0, scramble=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (n, d))
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)
        np.testing.assert_array_equal(result, np.array([[0., 0.],
                                                        [0.5, 0.5],
                                                        [0.75, 0.25],
                                                        [0.25, 0.75]]))

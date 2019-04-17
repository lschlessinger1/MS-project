from unittest import TestCase

import numpy as np
from numpy.linalg import LinAlgError

from src.autoks.distance.distance import fix_numerical_problem


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

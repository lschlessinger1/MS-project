from unittest import TestCase

import numpy as np
from GPy.kern import RBF

from src.autoks.acquisition_function import AcquisitionFunction, UniformScorer
from src.autoks.kernel import AKSKernel


class TestAcquisitionFunction(TestCase):

    def test_score(self):
        f_acq = AcquisitionFunction()
        kernel = AKSKernel(RBF(1))
        x_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([[5], [10]])
        self.assertRaises(NotImplementedError, f_acq.score, 0, [kernel], x_train, y_train)


class TestUniformScorer(TestCase):

    def test_score(self):
        f_acq = UniformScorer()
        kernel = AKSKernel(RBF(1))
        x_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([[5], [10]])
        result = f_acq.score(0, [kernel], x_train, y_train)
        self.assertTrue(isinstance(result, float) or isinstance(result, int))
        self.assertEqual(result, UniformScorer.CONST_SCORE)

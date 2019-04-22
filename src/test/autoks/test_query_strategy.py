from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.kern import RBF

from src.autoks.acquisition_function import AcquisitionFunction
from src.autoks.kernel import AKSKernel
from src.autoks.query_strategy import QueryStrategy, NaiveQueryStrategy, BestScoreStrategy


class TestQueryStrategy(TestCase):

    def setUp(self):
        self.kernels = [AKSKernel(RBF(1)), AKSKernel(RBF(1))]
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])

    def test_query(self):
        scoring_func = MagicMock(name='acq_func')
        qs = QueryStrategy(1, scoring_func)
        self.assertRaises(NotImplementedError, qs.query, [0], self.kernels, self.x_train, self.y_train)

    def test_score_kernels(self):
        score_const = 100
        scoring_func = MagicMock(name='acq_func')
        scoring_func.score.return_value = score_const
        qs = QueryStrategy(1, scoring_func)
        result = qs.score_kernels(list(range(len(self.kernels))), self.kernels, self.x_train, self.y_train)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.kernels))
        self.assertCountEqual(result, [score_const for _ in range(len(self.kernels))])
        self.assertEqual(scoring_func.score.call_count, len(self.kernels))


class TestNaiveQueryStrategy(TestCase):

    def setUp(self):
        self.kernels = [AKSKernel(RBF(1)), AKSKernel(RBF(1))]
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])
        self.qs = NaiveQueryStrategy()

    def test_query(self):
        # test all are selected and that scores are of some constant
        result = self.qs.query(list(range(len(self.kernels))), self.kernels, self.x_train, self.y_train)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertCountEqual(result[0].tolist(), [i for i in range(len(self.kernels))])
        self.assertIsInstance(result[1], list)
        self.assertTrue(all(isinstance(i, float) or isinstance(i, int) for i in result[1]))
        self.assertEqual(len(set(result[1])), 1)  # assert all items equal

    def test_select(self):
        result = self.qs.select(np.array(self.kernels), np.array([1, 2]))
        self.assertListEqual(list(result.tolist()), self.kernels)

    def test_arg_select(self):
        result = self.qs.arg_select(np.array(self.kernels), np.array([[1, 2]]))
        self.assertListEqual(list(result.tolist()), [i for i in range(len(self.kernels))])


class TestBestScoreStrategy(TestCase):

    def setUp(self):
        self.kernels = [AKSKernel(RBF(1)), AKSKernel(RBF(1))]
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])

    def test_query(self):
        scoring_func = AcquisitionFunction()
        scoring_func.score = MagicMock(spec=True)
        scores = [3, 20]
        scoring_func.score.side_effect = scores
        qs = BestScoreStrategy(scoring_func=scoring_func)
        result = qs.query(list(range(len(self.kernels))), self.kernels, self.x_train, self.y_train)
        self.assertEqual(len(scores), scoring_func.score.call_count)
        self.assertListEqual(list(result[0]), [1])
        self.assertListEqual(result[1], scores)

    def test_select(self):
        scoring_func = AcquisitionFunction()
        scores = [3, 20]
        qs = BestScoreStrategy(scoring_func=scoring_func)
        result = qs.select(np.array(self.kernels), np.array(scores))
        self.assertEqual(result, self.kernels[1])

    def test_arg_select(self):
        scoring_func = AcquisitionFunction()
        scores = [3, 20]
        qs = BestScoreStrategy(scoring_func=scoring_func)
        result = qs.arg_select(np.array(self.kernels), np.array(scores))
        self.assertEqual(result, [1])

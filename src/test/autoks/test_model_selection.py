from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.kern import RationalQuadratic, RBF, LinScaleShift

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import CKSGrammar
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.core.model_selection.boms_model_selector import BomsModelSelector
from src.autoks.core.model_selection.cks_model_selector import CKSModelSelector
from src.autoks.core.query_strategy import QueryStrategy


class TestModelSelector(TestCase):

    def setUp(self):
        self.gp_models = [GPModel(Covariance(RationalQuadratic(1))), GPModel(Covariance(RBF(1) + RBF(1))),
                          GPModel(Covariance(RBF(1)))]

        grammar = MagicMock()
        objective = MagicMock()
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])
        self.x_test = np.array([[10, 20, 30], [40, 50, 60]])
        self.y_test = np.array([[2], [1]])
        self.model_selector = ModelSelector(grammar, objective)

    def test_propose_new_models(self):
        expected = [Covariance(RBF(1)), Covariance(RationalQuadratic(1)), Covariance(RBF(1) + RationalQuadratic(1)),
                    Covariance(RBF(1) * RationalQuadratic(1))]
        self.model_selector.grammar.get_candidates.return_value = expected

        pop = ActiveModelPopulation()
        pop.update(self.gp_models)
        actual = self.model_selector.propose_new_models(pop)
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for expected_cov, actual_cov in zip(expected, actual):
            self.assertEqual(expected_cov.infix, actual_cov.covariance.infix)


class TestCKSModelSelector(TestCase):

    def setUp(self):
        self.se0 = Covariance(RBF(1, active_dims=[0]))
        self.se1 = Covariance(RBF(1, active_dims=[1]))
        self.se2 = Covariance(RBF(1, active_dims=[2]))
        self.rq0 = Covariance(RationalQuadratic(1, active_dims=[0]))
        self.rq1 = Covariance(RationalQuadratic(1, active_dims=[1]))
        self.rq2 = Covariance(RationalQuadratic(1, active_dims=[2]))
        self.lin0 = Covariance(LinScaleShift(1, active_dims=[0]))

    def test_get_initial_candidate_covariances(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        model_selector = CKSModelSelector(grammar)

        actual = model_selector.get_initial_candidate_covariances()
        expected = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for expected_cov, actual_cov in zip(expected, actual):
            self.assertEqual(expected_cov.infix, actual_cov.infix)

        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=1)
        model_selector = CKSModelSelector(grammar)

        actual = model_selector.get_initial_candidate_covariances()
        expected = [self.se0, self.rq0]
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for expected_cov, actual_cov in zip(expected, actual):
            self.assertEqual(expected_cov.infix, actual_cov.infix)


class TestBomsModelSelector(TestCase):

    def setUp(self):
        self.gp_models = [GPModel(Covariance(RationalQuadratic(1))), GPModel(Covariance(RBF(1) + RBF(1))),
                          GPModel(Covariance(RBF(1)))]

        grammar = MagicMock()
        kernel_selector = MagicMock()
        objective = MagicMock()
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])
        self.x_test = np.array([[10, 20, 30], [40, 50, 60]])
        self.y_test = np.array([[2], [1]])
        self.model_selector = BomsModelSelector(grammar, kernel_selector, objective)

    def test_query_kernels(self):
        scoring_func = MagicMock(name='acq_func')
        scores = [10, 20, 30]
        scoring_func.score.side_effect = scores
        qs = QueryStrategy(1, scoring_func)
        qs.arg_select = MagicMock()
        ind = [1, 2]
        qs.arg_select.return_value = ind

        pop = GPModelPopulation()
        pop.update(self.gp_models)
        result = self.model_selector.query_models(pop, qs, self.x_train, self.y_train)
        self.assertListEqual([self.gp_models[i] for i in ind], result[0].tolist())
        self.assertListEqual(ind, result[1])
        self.assertListEqual(scores, result[2])

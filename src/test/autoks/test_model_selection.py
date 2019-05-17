from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.kern import RationalQuadratic, RBF

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.model_selection import ModelSelector
from src.autoks.core.query_strategy import QueryStrategy


class TestModelSelector(TestCase):

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
        self.model_selector = ModelSelector(grammar, kernel_selector, objective)

    def test_query_kernels(self):
        scoring_func = MagicMock(name='acq_func')
        scores = [10, 20, 30]
        scoring_func.score.side_effect = scores
        qs = QueryStrategy(1, scoring_func)
        qs.arg_select = MagicMock()
        ind = [1, 2]
        qs.arg_select.return_value = ind
        result = self.model_selector.query_models(self.gp_models, qs, self.x_train, self.y_train)
        self.assertListEqual([self.gp_models[i] for i in ind], result[0].tolist())
        self.assertListEqual(ind, result[1])
        self.assertListEqual(scores, result[2])

    def test_select_parents(self):
        parents = self.gp_models[:2]
        self.model_selector.kernel_selector.select_parents.return_value = parents
        result = self.model_selector.select_parents(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, parents)

    def test_propose_new_kernels(self):
        expansion = [GPModel(kern) for kern in [Covariance(RBF(1)), Covariance(RationalQuadratic(1)),
                                                Covariance(RBF(1) + RationalQuadratic(1)), Covariance(RBF(1)
                                                                                                      * RationalQuadratic(
                1))]]
        self.model_selector.grammar.get_candidates.return_value = expansion
        result = self.model_selector.propose_new_kernels(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expansion)

    def test_select_offspring(self):
        offspring = self.gp_models[:2]
        self.model_selector.kernel_selector.select_offspring.return_value = offspring
        result = self.model_selector.select_offspring(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, offspring)
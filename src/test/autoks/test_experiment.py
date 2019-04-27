from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, call

import numpy as np
from GPy.kern import RationalQuadratic, RBF

from src.autoks.core.covariance import Covariance
from src.autoks.core.experiment import Experiment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.query_strategy import QueryStrategy


class TestExperiment(TestCase):

    def setUp(self):
        self.gp_models = [GPModel(Covariance(RationalQuadratic(1))), GPModel(Covariance(RBF(1))),
                          GPModel(Covariance(RBF(1)))]

        grammar = MagicMock()
        kernel_selector = MagicMock()
        objective = MagicMock()
        x_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([[5], [10]])
        x_test = np.array([[10, 20, 30], [40, 50, 60]])
        y_test = np.array([[2], [1]])
        self.exp = Experiment(grammar, kernel_selector, objective, x_train, y_train, x_test, y_test,
                              use_surrogate=False)

    def test_kernel_search(self):
        max_depth = 2
        self.exp.max_depth = max_depth
        initial_kernels = self.gp_models
        self.exp.grammar.initialize.return_value = initial_kernels

        def return_arg(arg):
            return arg

        self.exp.optimize_model = MagicMock()
        self.exp.optimize_model.side_effect = return_arg

        def get_score(*args, **kwargs):
            return 1

        self.exp.objective.side_effect = get_score

        def get_candidates(parents, *args, **kwargs):
            new_kernels = []
            for parent in parents:
                new_kernels.append(parent)
                new_kernels.append(GPModel(Covariance(parent.covariance.raw_kernel + RBF(1))))
                new_kernels.append(GPModel(Covariance(parent.covariance.raw_kernel + RationalQuadratic(1))))
                new_kernels.append(GPModel(Covariance(parent.covariance.raw_kernel * RBF(1))))
                new_kernels.append(GPModel(Covariance(parent.covariance.raw_kernel * RationalQuadratic(1))))
            return new_kernels

        self.exp.grammar.get_candidates.side_effect = get_candidates

        scoring_func = MagicMock(name='acq_func')
        scores = [10, 20, 30]
        scoring_func.score.side_effect = scores
        qs_init = QueryStrategy(1, scoring_func)
        qs_init.arg_select = MagicMock()
        ind_init = [2]
        qs_init.arg_select.return_value = ind_init
        self.exp.init_query_strat = qs_init

        scoring_func = MagicMock(name='acq_func')
        score = 44
        scoring_func.score.return_value = score
        qs = QueryStrategy(1, scoring_func)
        qs.arg_select = MagicMock()
        ind = [0]  # always pick first unscored to evaluate
        qs.arg_select.return_value = ind
        self.exp.query_strat = qs

        def first_kernel(kernels: List[GPModel], _):
            return [] if len(kernels) == 0 else [kernels[0]]

        def all_but_first(kernels: List[GPModel], _):
            return kernels[1:]

        self.exp.kernel_selector.select_parents.side_effect = first_kernel
        self.exp.kernel_selector.select_offspring.side_effect = all_but_first

        result = self.exp.model_search()

        self.exp.grammar.initialize.assert_called_once()

        # test that the first query gp_models is called with initial gp_models and iniital query strat
        self.assertEqual(self.exp.init_query_strat.scoring_func.score.call_count, 3)
        self.exp.init_query_strat.arg_select.assert_called()

        n_expansions = max_depth + 1
        n_optimizations = len(ind_init) + len(ind) * n_expansions
        n_evals = n_optimizations
        self.assertEqual(n_optimizations, self.exp.optimize_model.call_count)
        self.assertEqual(n_evals, self.exp.n_evals)
        self.assertEqual(n_expansions, self.exp.grammar.get_candidates.call_count)

        self.assertEqual(n_expansions, self.exp.kernel_selector.select_parents.call_count)
        self.assertEqual(n_expansions, self.exp.kernel_selector.select_offspring.call_count)

        self.assertLessEqual(self.exp.n_evals, self.exp.eval_budget)
        self.assertLessEqual(self.exp.max_depth, self.exp.grammar.get_candidates.call_count)
        self.assertLessEqual(self.exp.max_depth, self.exp.grammar.get_candidates.call_count)
        self.assertIsInstance(result, list)

    def test_query_kernels(self):
        scoring_func = MagicMock(name='acq_func')
        scores = [10, 20, 30]
        scoring_func.score.side_effect = scores
        qs = QueryStrategy(1, scoring_func)
        qs.arg_select = MagicMock()
        ind = [1, 2]
        qs.arg_select.return_value = ind
        result = self.exp.query_models(self.gp_models, qs)
        self.assertListEqual([self.gp_models[i] for i in ind], result[0].tolist())
        self.assertListEqual(ind, result[1])
        self.assertListEqual(scores, result[2])

    def test_select_parents(self):
        parents = self.gp_models[:2]
        self.exp.kernel_selector.select_parents.return_value = parents
        result = self.exp.select_parents(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, parents)

    def test_propose_new_kernels(self):
        expansion = [GPModel(kern) for kern in [Covariance(RBF(1)), Covariance(RationalQuadratic(1)),
                                                Covariance(RBF(1) + RationalQuadratic(1)), Covariance(RBF(1)
                                                                                                      * RationalQuadratic(
                1))]]
        self.exp.grammar.get_candidates.return_value = expansion
        result = self.exp.propose_new_kernels(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expansion)

    def test_select_offspring(self):
        offspring = self.gp_models[:2]
        self.exp.kernel_selector.select_offspring.return_value = offspring
        result = self.exp.select_offspring(self.gp_models)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, offspring)

    def test_opt_and_eval_kernels(self):
        # test that opt and eval is called for all gp_models
        self.exp.optimize_model = MagicMock()
        self.exp.evaluate_model = MagicMock()
        self.exp.update_stats = MagicMock()

        self.exp.opt_and_eval_models(self.gp_models)

        self.assertEqual(len(self.gp_models), self.exp.optimize_model.call_count)
        self.assertEqual(len(self.gp_models), self.exp.evaluate_model.call_count)
        self.exp.optimize_model.assert_has_calls([call(k) for k in self.gp_models], any_order=True)
        # self.exp.evaluate_model.assert_has_calls([call(k) for k in self.gp_models], any_order=True)

    def test_optimize_kernel(self):
        self.exp.gp_model = MagicMock()
        self.exp.optimize_model(self.gp_models[0])
        self.assertEqual(1, self.exp.gp_model.optimize.call_count)
        self.assertEqual(1, self.exp.gp_model.optimize_restarts.call_count)

    def test_evaluate_kernel(self):
        score = 10
        self.exp.objective.return_value = score
        kern = self.gp_models[0]
        self.exp.evaluate_model(kern)
        self.exp.objective.assert_called_once_with(self.exp.gp_model)
        self.assertEqual(kern.score, score)
        self.assertFalse(kern.nan_scored)
        self.assertTrue(kern.evaluated)

    def test_remove_nan_scored_kernels(self):
        kernels = [GPModel(Covariance(RationalQuadratic(1)), nan_scored=True), GPModel(RBF(1)),
                   GPModel(Covariance(RBF(1)), nan_scored=True)]
        result = self.exp.remove_nan_scored_models(kernels)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, [kernels[1]])

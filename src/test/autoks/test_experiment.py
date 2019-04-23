from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, call

import numpy as np
from GPy.kern import RationalQuadratic, RBF

from src.autoks.experiment import Experiment
from src.autoks.kernel import AKSKernel
from src.autoks.query_strategy import QueryStrategy


class TestExperiment(TestCase):

    def setUp(self):
        self.kernels = [AKSKernel(RationalQuadratic(1)), AKSKernel(RBF(1)), AKSKernel(RBF(1))]

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
        initial_kernels = self.kernels
        self.exp.grammar.initialize.return_value = initial_kernels

        def return_arg(arg):
            return arg

        self.exp.optimize_kernel = MagicMock()
        self.exp.optimize_kernel.side_effect = return_arg

        def get_score(*args, **kwargs):
            return 1

        self.exp.objective.side_effect = get_score

        def expand(parents, *args, **kwargs):
            new_kernels = []
            for parent in parents:
                new_kernels.append(parent)
                new_kernels.append(AKSKernel(parent.kernel + RBF(1)))
                new_kernels.append(AKSKernel(parent.kernel + RationalQuadratic(1)))
                new_kernels.append(AKSKernel(parent.kernel * RBF(1)))
                new_kernels.append(AKSKernel(parent.kernel * RationalQuadratic(1)))
            return new_kernels

        self.exp.grammar.expand.side_effect = expand

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

        def first_kernel(kernels: List[AKSKernel], _):
            return [] if len(kernels) == 0 else [kernels[0]]

        def all_but_first(kernels: List[AKSKernel], _):
            return kernels[1:]

        self.exp.kernel_selector.select_parents.side_effect = first_kernel
        self.exp.kernel_selector.select_offspring.side_effect = all_but_first
        self.exp.kernel_selector.prune_candidates.side_effect = all_but_first

        result = self.exp.kernel_search()

        self.exp.grammar.initialize.assert_called_once()

        # test that the first query kernels is called with initial kernels and iniital query strat
        self.assertEqual(self.exp.init_query_strat.scoring_func.score.call_count, 3)
        self.exp.init_query_strat.arg_select.assert_called()

        n_expansions = max_depth + 1
        n_optimizations = len(ind_init) + len(ind) * n_expansions
        n_evals = n_optimizations
        self.assertEqual(n_optimizations, self.exp.optimize_kernel.call_count)
        self.assertEqual(n_evals, self.exp.n_evals)
        self.assertEqual(n_expansions, self.exp.grammar.expand.call_count)

        self.assertEqual(n_expansions, self.exp.kernel_selector.select_parents.call_count)
        self.assertEqual(n_expansions, self.exp.kernel_selector.select_offspring.call_count)
        self.assertEqual(n_expansions, self.exp.kernel_selector.prune_candidates.call_count)

        self.assertLessEqual(self.exp.n_evals, self.exp.eval_budget)
        self.assertLessEqual(self.exp.max_depth, self.exp.grammar.expand.call_count)
        self.assertIsInstance(result, list)

    def test_query_kernels(self):
        scoring_func = MagicMock(name='acq_func')
        scores = [10, 20, 30]
        scoring_func.score.side_effect = scores
        qs = QueryStrategy(1, scoring_func)
        qs.arg_select = MagicMock()
        ind = [1, 2]
        qs.arg_select.return_value = ind
        result = self.exp.query_kernels(self.kernels, qs)
        self.assertListEqual([self.kernels[i] for i in ind], result[0].tolist())
        self.assertListEqual(ind, result[1])
        self.assertListEqual(scores, result[2])

    def test_select_parents(self):
        parents = self.kernels[:2]
        self.exp.kernel_selector.select_parents.return_value = parents
        result = self.exp.select_parents(self.kernels)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, parents)

    def test_propose_new_kernels(self):
        expansion = [AKSKernel(kern) for kern in [RBF(1), RationalQuadratic(1), RBF(1) + RationalQuadratic(1), RBF(1)
                                                  * RationalQuadratic(1)]]
        self.exp.grammar.expand.return_value = expansion
        result = self.exp.propose_new_kernels(self.kernels)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expansion)

    def test_prune_kernels(self):
        ind = [0, 1]
        pruned = [self.kernels[i] for i in ind]
        self.exp.kernel_selector.prune_candidates.return_value = pruned
        result = self.exp.prune_kernels(self.kernels, [0.1, 0.2, 0.3], ind)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, pruned)

    def test_select_offspring(self):
        offspring = self.kernels[:2]
        self.exp.kernel_selector.select_offspring.return_value = offspring
        result = self.exp.select_offspring(self.kernels)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, offspring)

    def test_opt_and_eval_kernels(self):
        # test that opt and eval is called for all kernels
        self.exp.optimize_kernel = MagicMock()
        self.exp.evaluate_kernel = MagicMock()
        self.exp.update_stats = MagicMock()

        self.exp.opt_and_eval_kernels(self.kernels)

        self.assertEqual(len(self.kernels), self.exp.optimize_kernel.call_count)
        self.assertEqual(len(self.kernels), self.exp.evaluate_kernel.call_count)
        self.exp.optimize_kernel.assert_has_calls([call(k) for k in self.kernels], any_order=True)
        # self.exp.evaluate_kernel.assert_has_calls([call(k) for k in self.kernels], any_order=True)

    def test_optimize_kernel(self):
        self.exp.gp_model = MagicMock()
        self.exp.optimize_kernel(self.kernels[0])
        self.assertEqual(1, self.exp.gp_model.optimize.call_count)
        self.assertEqual(1, self.exp.gp_model.optimize_restarts.call_count)

    def test_evaluate_kernel(self):
        score = 10
        self.exp.objective.return_value = score
        kern = self.kernels[0]
        self.exp.evaluate_kernel(kern)
        self.exp.objective.assert_called_once_with(self.exp.gp_model)
        self.assertEqual(kern.score, score)
        self.assertFalse(kern.nan_scored)
        self.assertTrue(kern.evaluated)

    def test_remove_nan_scored_kernels(self):
        kernels = [AKSKernel(RationalQuadratic(1), nan_scored=True), AKSKernel(RBF(1)),
                   AKSKernel(RBF(1), nan_scored=True)]
        result = self.exp.remove_nan_scored_kernels(kernels)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, [kernels[1]])

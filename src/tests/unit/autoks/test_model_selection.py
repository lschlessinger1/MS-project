from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from GPy.kern import RationalQuadratic, RBF, LinScaleShift

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import ActiveModelPopulation
from src.autoks.core.grammar import CKSGrammar, RandomGrammar
from src.autoks.core.model_selection import EvolutionaryModelSelector
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.core.model_selection.boms_model_selector import BomsModelSelector
from src.autoks.core.model_selection.cks_model_selector import CKSModelSelector
from src.evalg.serialization import Serializable


class TestModelSelector(TestCase):

    def setUp(self):
        self.gp_models = [GPModel(Covariance(RationalQuadratic(1))), GPModel(Covariance(RBF(1) + RBF(1))),
                          GPModel(Covariance(RBF(1)))]

        grammar = RandomGrammar()
        grammar.build(n_dims=1)

        fitness_fn = 'nbic'
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])
        self.x_test = np.array([[10, 20, 30], [40, 50, 60]])
        self.y_test = np.array([[2], [1]])
        self.model_selector = ModelSelector(grammar, fitness_fn)

    @patch('src.autoks.core.grammar.BaseGrammar.get_candidates')
    def test_propose_new_models(self, mock_get_candidates):
        expected = [Covariance(RBF(1)), Covariance(RationalQuadratic(1)), Covariance(RBF(1) + RationalQuadratic(1)),
                    Covariance(RBF(1) * RationalQuadratic(1))]
        mock_get_candidates.return_value = expected

        pop = ActiveModelPopulation()
        pop.update(self.gp_models)
        actual = self.model_selector._propose_new_models(pop, callbacks=MagicMock())
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for expected_cov, actual_cov in zip(expected, actual):
            self.assertEqual(expected_cov.infix, actual_cov.covariance.infix)

    def test_to_dict(self):
        test_cases = (
            False, True
        )
        for built in test_cases:
            with self.subTest(name=built):
                if built:
                    self.model_selector._prepare_data(self.x_train, self.y_train)

                actual = self.model_selector.to_dict()

                self.assertIsInstance(actual, dict)

                self.assertIn('grammar', actual)
                self.assertIn('fitness_fn', actual)
                self.assertIn('n_parents', actual)
                self.assertIn('n_evals', actual)
                self.assertIn('additive_form', actual)
                self.assertIn('optimizer', actual)
                self.assertIn('n_restarts_optimizer', actual)
                self.assertIn('standardize_x', actual)
                self.assertIn('standardize_y', actual)
                self.assertIn('total_eval_time', actual)
                self.assertIn('total_expansion_time', actual)
                self.assertIn('total_model_search_time', actual)
                self.assertIn('gp_fn_name', actual)
                self.assertIn('gp_args', actual)
                self.assertIn('name', actual)
                self.assertIn('built', actual)
                self.assertIn('selected_models', actual)
                self.assertIn('_x_train_mean', actual)
                self.assertIn('_x_train_std', actual)
                self.assertIn('_y_train_mean', actual)
                self.assertIn('_y_train_std', actual)

                self.assertEqual(self.model_selector.grammar.to_dict(), actual['grammar'])
                self.assertEqual(self.model_selector.fitness_fn_name, actual['fitness_fn'])
                self.assertEqual(self.model_selector.n_parents, actual['n_parents'])
                self.assertEqual(self.model_selector.n_evals, actual['n_evals'])
                self.assertEqual(self.model_selector.additive_form, actual['additive_form'])
                self.assertEqual(self.model_selector.optimizer, actual['optimizer'])
                self.assertEqual(self.model_selector.n_restarts_optimizer, actual['n_restarts_optimizer'])
                self.assertEqual(self.model_selector.standardize_x, actual['standardize_x'])
                self.assertEqual(self.model_selector.standardize_y, actual['standardize_y'])
                self.assertEqual(self.model_selector.total_eval_time, actual['total_eval_time'])
                self.assertEqual(self.model_selector.total_expansion_time, actual['total_expansion_time'])
                self.assertEqual(self.model_selector.total_model_search_time, actual['total_model_search_time'])
                self.assertEqual(self.model_selector._gp_fn_name, actual['gp_fn_name'])
                self.assertEqual(self.model_selector.name, actual['name'])
                self.assertEqual(self.model_selector.built, actual['built'])
                expected_selected_models = [m.to_dict() for m in self.model_selector.selected_models]
                actual_selected_models = [m.to_dict() for m in actual['selected_models']]
                self.assertEqual(expected_selected_models, actual_selected_models)
                if not built:
                    self.assertEqual(self.model_selector._x_train_mean, actual['_x_train_mean'])
                    self.assertEqual(self.model_selector._x_train_std, actual['_x_train_std'])
                    self.assertEqual(self.model_selector._y_train_mean, actual['_y_train_mean'])
                    self.assertEqual(self.model_selector._y_train_std, actual['_y_train_std'])
                else:
                    self.assertEqual(self.model_selector._x_train_mean.tolist(), actual['_x_train_mean'])
                    self.assertEqual(self.model_selector._x_train_std.tolist(), actual['_x_train_std'])
                    self.assertEqual(self.model_selector._y_train_mean.tolist(), actual['_y_train_mean'])
                    self.assertEqual(self.model_selector._y_train_std.tolist(), actual['_y_train_std'])

    def test_from_dict(self):
        test_cases_built = (False, True)
        for built in test_cases_built:
            with self.subTest(built=built):
                test_cases_cls = (ModelSelector, Serializable)
                for cls in test_cases_cls:
                    with self.subTest(cls=cls):
                        if built:
                            self.model_selector._prepare_data(self.x_train, self.y_train)

                        actual = cls.from_dict(self.model_selector.to_dict())

                        self.assertIsInstance(actual, ModelSelector)

                        self.assertEqual(self.model_selector.grammar.__class__.__name__,
                                         actual.grammar.__class__.__name__)
                        self.assertEqual(self.model_selector.fitness_fn_name, actual.fitness_fn_name)
                        self.assertEqual(self.model_selector.fitness_fn, actual.fitness_fn)
                        self.assertEqual(self.model_selector.n_parents, actual.n_parents)
                        self.assertEqual(self.model_selector.n_evals, actual.n_evals)
                        self.assertEqual(self.model_selector.additive_form, actual.additive_form)
                        self.assertEqual(self.model_selector.optimizer, actual.optimizer)
                        self.assertEqual(self.model_selector.n_restarts_optimizer, actual.n_restarts_optimizer)
                        self.assertEqual(self.model_selector.standardize_x, actual.standardize_x)
                        self.assertEqual(self.model_selector.standardize_y, actual.standardize_y)
                        self.assertEqual(self.model_selector.total_eval_time, actual.total_eval_time)
                        self.assertEqual(self.model_selector.total_expansion_time, actual.total_expansion_time)
                        self.assertEqual(self.model_selector.total_model_search_time, actual.total_model_search_time)
                        self.assertEqual(self.model_selector._gp_fn_name, actual._gp_fn_name)
                        self.assertEqual(self.model_selector._gp_args, actual._gp_args)
                        self.assertEqual(self.model_selector.name, actual.name)
                        self.assertEqual(self.model_selector.built, actual.built)
                        self.assertEqual(self.model_selector.selected_models, actual.selected_models)
                        if not built:
                            self.assertIsNone(actual._x_train_mean)
                            self.assertIsNone(actual._x_train_std)
                            self.assertIsNone(actual._y_train_mean)
                            self.assertIsNone(actual._y_train_std)
                        else:
                            self.assertEqual(self.model_selector._x_train_mean.tolist(), actual._x_train_mean.tolist())
                            self.assertEqual(self.model_selector._x_train_std.tolist(), actual._x_train_std.tolist())
                            self.assertEqual(self.model_selector._y_train_mean.tolist(), actual._y_train_mean.tolist())
                            self.assertEqual(self.model_selector._y_train_std.tolist(), actual._y_train_std.tolist())


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
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)

        model_selector = CKSModelSelector(grammar)

        actual = model_selector._get_initial_candidate_covariances()
        expected = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for expected_cov, actual_cov in zip(expected, actual):
            self.assertEqual(expected_cov.infix, actual_cov.infix)

        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=1)

        model_selector = CKSModelSelector(grammar)

        actual = model_selector._get_initial_candidate_covariances()
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


class TestEvolutionaryModelSelector(TestCase):

    def setUp(self) -> None:
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])

        self.x_test = np.array([[10, 20, 30], [40, 50, 60]])
        self.y_test = np.array([[2], [1]])

        self.model_selector = EvolutionaryModelSelector()

    def test_to_dict(self):
        test_cases = (
            False, True
        )
        for built in test_cases:
            with self.subTest(name=built):
                if built:
                    self.model_selector._prepare_data(self.x_train, self.y_train)
                actual = self.model_selector.to_dict()

                self.assertIsInstance(actual, dict)

                self.assertIn('grammar', actual)
                self.assertIn('fitness_fn', actual)
                self.assertIn('n_parents', actual)
                self.assertIn('n_evals', actual)
                self.assertIn('additive_form', actual)
                self.assertIn('optimizer', actual)
                self.assertIn('n_restarts_optimizer', actual)
                self.assertIn('standardize_x', actual)
                self.assertIn('standardize_y', actual)
                self.assertIn('total_eval_time', actual)
                self.assertIn('total_expansion_time', actual)
                self.assertIn('total_model_search_time', actual)
                self.assertIn('gp_fn_name', actual)
                self.assertIn('gp_args', actual)
                self.assertIn('name', actual)
                self.assertIn('built', actual)
                self.assertIn('selected_models', actual)
                self.assertIn('_x_train_mean', actual)
                self.assertIn('_x_train_std', actual)
                self.assertIn('_y_train_mean', actual)
                self.assertIn('_y_train_std', actual)
                self.assertIn('initializer', actual)
                self.assertIn('n_init_trees', actual)
                self.assertIn('max_offspring', actual)
                self.assertIn('fitness_sharing', actual)

                self.assertEqual(self.model_selector.grammar.to_dict(), actual['grammar'])
                self.assertEqual(self.model_selector.fitness_fn_name, actual['fitness_fn'])
                self.assertEqual(self.model_selector.n_parents, actual['n_parents'])
                self.assertEqual(self.model_selector.n_evals, actual['n_evals'])
                self.assertEqual(self.model_selector.additive_form, actual['additive_form'])
                self.assertEqual(self.model_selector.optimizer, actual['optimizer'])
                self.assertEqual(self.model_selector.n_restarts_optimizer, actual['n_restarts_optimizer'])
                self.assertEqual(self.model_selector.standardize_x, actual['standardize_x'])
                self.assertEqual(self.model_selector.standardize_y, actual['standardize_y'])
                self.assertEqual(self.model_selector.total_eval_time, actual['total_eval_time'])
                self.assertEqual(self.model_selector.total_expansion_time, actual['total_expansion_time'])
                self.assertEqual(self.model_selector.total_model_search_time, actual['total_model_search_time'])
                self.assertEqual(self.model_selector._gp_fn_name, actual['gp_fn_name'])
                self.assertEqual(self.model_selector.name, actual['name'])
                self.assertEqual(self.model_selector.built, actual['built'])
                expected_selected_models = [m.to_dict() for m in self.model_selector.selected_models]
                actual_selected_models = [m.to_dict() for m in actual['selected_models']]
                self.assertEqual(expected_selected_models, actual_selected_models)
                if not built:
                    self.assertEqual(self.model_selector._x_train_mean, actual['_x_train_mean'])
                    self.assertEqual(self.model_selector._x_train_std, actual['_x_train_std'])
                    self.assertEqual(self.model_selector._y_train_mean, actual['_y_train_mean'])
                    self.assertEqual(self.model_selector._y_train_std, actual['_y_train_std'])
                else:
                    self.assertEqual(self.model_selector._x_train_mean.tolist(), actual['_x_train_mean'])
                    self.assertEqual(self.model_selector._x_train_std.tolist(), actual['_x_train_std'])
                    self.assertEqual(self.model_selector._y_train_mean.tolist(), actual['_y_train_mean'])
                    self.assertEqual(self.model_selector._y_train_std.tolist(), actual['_y_train_std'])
                self.assertEqual(self.model_selector.initializer.to_dict(), actual['initializer'])
                self.assertEqual(self.model_selector.n_init_trees, actual['n_init_trees'])
                self.assertEqual(self.model_selector.max_offspring, actual['max_offspring'])
                self.assertEqual(self.model_selector.fitness_sharing, actual['fitness_sharing'])

    def test_from_dict_unbuilt(self):
        test_cases_built = (False, True)
        for built in test_cases_built:
            with self.subTest(built=built):
                test_cases_cls = (ModelSelector, Serializable)
                for cls in test_cases_cls:
                    with self.subTest(cls=cls):
                        if built:
                            self.model_selector._prepare_data(self.x_train, self.y_train)
                        actual = cls.from_dict(self.model_selector.to_dict())

                        self.assertIsInstance(actual, EvolutionaryModelSelector)

                        self.assertEqual(self.model_selector.grammar.__class__.__name__,
                                         actual.grammar.__class__.__name__)
                        self.assertEqual(self.model_selector.fitness_fn_name, actual.fitness_fn_name)
                        self.assertEqual(self.model_selector.fitness_fn, actual.fitness_fn)
                        self.assertEqual(self.model_selector.n_parents, actual.n_parents)
                        self.assertEqual(self.model_selector.n_evals, actual.n_evals)
                        self.assertEqual(self.model_selector.additive_form, actual.additive_form)
                        self.assertEqual(self.model_selector.optimizer, actual.optimizer)
                        self.assertEqual(self.model_selector.n_restarts_optimizer, actual.n_restarts_optimizer)
                        self.assertEqual(self.model_selector.standardize_x, actual.standardize_x)
                        self.assertEqual(self.model_selector.standardize_y, actual.standardize_y)
                        self.assertEqual(self.model_selector.total_eval_time, actual.total_eval_time)
                        self.assertEqual(self.model_selector.total_expansion_time, actual.total_expansion_time)
                        self.assertEqual(self.model_selector.total_model_search_time, actual.total_model_search_time)
                        self.assertEqual(self.model_selector._gp_fn_name, actual._gp_fn_name)
                        self.assertEqual(self.model_selector._gp_args, actual._gp_args)
                        self.assertEqual(self.model_selector.name, actual.name)
                        self.assertEqual(self.model_selector.built, actual.built)
                        self.assertEqual(self.model_selector.selected_models, actual.selected_models)
                        if not built:
                            self.assertIsNone(actual._x_train_mean)
                            self.assertIsNone(actual._x_train_std)
                            self.assertIsNone(actual._y_train_mean)
                            self.assertIsNone(actual._y_train_std)
                        else:
                            self.assertEqual(self.model_selector._x_train_mean.tolist(), actual._x_train_mean.tolist())
                            self.assertEqual(self.model_selector._x_train_std.tolist(), actual._x_train_std.tolist())
                            self.assertEqual(self.model_selector._y_train_mean.tolist(), actual._y_train_mean.tolist())
                            self.assertEqual(self.model_selector._y_train_std.tolist(), actual._y_train_std.tolist())
                        self.assertEqual(self.model_selector.initializer.__class__.__name__,
                                         actual.initializer.__class__.__name__)
                        self.assertEqual(self.model_selector.n_init_trees, actual.n_init_trees)
                        self.assertEqual(self.model_selector.max_offspring, actual.max_offspring)
                        self.assertEqual(self.model_selector.fitness_sharing, actual.fitness_sharing)

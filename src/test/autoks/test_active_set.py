from unittest import TestCase

from GPy.kern import RBF, RationalQuadratic

from src.autoks.active_set import ActiveSet
from src.autoks.core.gp_model import GPModel


class TestActiveSet(TestCase):

    def setUp(self) -> None:
        self.active_set = ActiveSet(max_n_models=5)

    def test_set_bad_remove_priority(self):
        with self.assertRaises(ValueError):
            remove_priority = [0, 1, 2, 3, 4, 0]
            self.active_set.remove_priority = remove_priority
        with self.assertRaises(ValueError):
            remove_priority = [10]
            self.active_set.remove_priority = remove_priority

    def test_get_index_to_insert_empty(self):
        expected_index = 0
        actual_index = self.active_set.get_index_to_insert()
        self.assertEqual(expected_index, actual_index)

    def test_get_index_to_insert_one_item(self):
        # add one model
        self.active_set.add_model(GPModel(RBF(1)))

        expected_index = 1
        actual_index = self.active_set.get_index_to_insert()
        self.assertEqual(expected_index, actual_index)

    def test_get_index_to_insert_full_no_priority(self):
        # add five models
        self.active_set.add_model(GPModel(RBF(1)))
        self.active_set.add_model(GPModel(RationalQuadratic(1)))
        self.active_set.add_model(GPModel(RBF(1)))
        self.active_set.add_model(GPModel(RationalQuadratic(1)))
        self.active_set.add_model(GPModel(RBF(1)))

        self.assertRaises(ValueError, self.active_set.get_index_to_insert)

    def test_get_index_to_insert_full_with_priority(self):
        # add five models
        self.active_set.add_model(GPModel(RBF(1)))
        self.active_set.add_model(GPModel(RationalQuadratic(1)))
        self.active_set.add_model(GPModel(RBF(1)))
        self.active_set.add_model(GPModel(RationalQuadratic(1)))
        self.active_set.add_model(GPModel(RBF(1)))

        remove_priority = [2]
        self.active_set.remove_priority = remove_priority
        actual = self.active_set.get_index_to_insert()
        expected = 2
        self.assertEqual(expected, actual)
        self.assertEqual(self.active_set.remove_priority, [])

        remove_priority = [0, 2, 3]
        self.active_set.remove_priority = remove_priority
        actual = self.active_set.get_index_to_insert()
        expected = 0
        self.assertEqual(expected, actual)
        self.assertEqual(self.active_set.remove_priority, [2, 3])

    def test_add_model_empty(self):
        candidate = GPModel(RBF(1))
        expected_ind, expected_status = 0, True
        actual_ind, actual_status = self.active_set.add_model(candidate)
        self.assertEqual(expected_ind, actual_ind)
        self.assertEqual(expected_status, actual_status)

        expected_models = [candidate, None, None, None, None]
        self.assertListEqual(expected_models, self.active_set.models)

        expected_next_ind = 1
        self.assertEqual(expected_next_ind, self.active_set.get_index_to_insert())

    def test_add_model_same(self):
        # TODO: should this actually be same kernel?
        models = [GPModel(RBF(1))] * 2
        self.active_set.add_model(models[0])

        expected_ind = -1
        expected_status = False
        actual_ind, actual_status = self.active_set.add_model(models[1])
        self.assertEqual(expected_ind, actual_ind)
        self.assertEqual(expected_status, actual_status)

        expected_models = [models[0], None, None, None, None]
        self.assertListEqual(expected_models, self.active_set.models)

        expected_next_ind = 1
        self.assertEqual(expected_next_ind, self.active_set.get_index_to_insert())

    def test_update_empty(self):
        candidates = [GPModel(RBF(1)), GPModel(RationalQuadratic(1))]
        expected_candidates_ind = [0, 1]
        new_candidates_ind = self.active_set.update(candidates)
        self.assertEqual(expected_candidates_ind, new_candidates_ind)

        expected_models = [candidates[0], candidates[1], None, None, None]
        self.assertListEqual(expected_models, self.active_set.models)

        expected_next_ind = 2
        self.assertEqual(expected_next_ind, self.active_set.get_index_to_insert())

    def test_update_exceed_max_no_remove(self):
        candidates = [GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)),
                      GPModel(RBF(1))]
        self.assertRaises(ValueError, self.active_set.update, candidates)

    def test_update_exceed_max_remove_set(self):
        candidates = [GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)), GPModel(RBF(1)),
                      GPModel(RationalQuadratic(1))]
        self.active_set.remove_priority = [0, 1, 2, 3, 4]
        actual_new_candidates_ind = self.active_set.update(candidates)
        expected_new_candidates_ind = [1, 2, 3, 4, 0]
        self.assertListEqual(expected_new_candidates_ind, actual_new_candidates_ind)

        expected_models = [candidates[5], candidates[1], candidates[2], candidates[3], candidates[4]]
        self.assertListEqual(expected_models, self.active_set.models)

        expected_next_ind = 1
        self.assertEqual(expected_next_ind, self.active_set.get_index_to_insert())

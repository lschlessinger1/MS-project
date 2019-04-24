import unittest

import numpy as np
from GPy.kern import RBF, RationalQuadratic, Add, LinScaleShift, Kern
from GPy.kern.src.kern import CombinationKernel

from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import BaseGrammar, CKSGrammar, remove_duplicate_kernels, BOMSGrammar
from src.test.autoks.support.util import has_combo_kernel_type, base_kernel_eq


class TestGrammar(unittest.TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RationalQuadratic(1, active_dims=[0])
        self.rq1 = RationalQuadratic(1, active_dims=[1])

    def test_remove_duplicate_kernels(self):
        kernels = [self.se0 + self.se0, self.se1, self.se0, self.se0, self.se1 + self.se0, self.se0 + self.se1]
        kernels_pruned = remove_duplicate_kernels(kernels)
        # should be SE0 + SE0, SE1, SE0, SE1 + SE0
        kernel_types_outer = [(Add, [0]), (RBF, [1]), (RBF, [0]), (Add, [0, 1])]
        for (k_class, dims), part in zip(kernel_types_outer, kernels_pruned):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims.tolist(), dims)

        kernel_types_inner_1 = [(RBF, 0), (RBF, 0)]
        sum_1 = kernels_pruned[0]
        for (k_class, dim), part in zip(kernel_types_inner_1, sum_1.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_2 = [(RBF, 1), (RBF, 0)]
        sum_2 = kernels_pruned[3]
        for (k_class, dim), part in zip(kernel_types_inner_2, sum_2.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)


class TestBaseGrammar(unittest.TestCase):

    def setUp(self):
        self.grammar = BaseGrammar(['SE', 'RQ'], 1)

    def test_initialize(self):
        self.assertRaises(NotImplementedError, self.grammar.initialize)

    def test_expand(self):
        seed_kernel = GPModel(RBF(1))
        self.assertRaises(NotImplementedError, self.grammar.expand, seed_kernel)


class TestCKSGrammar(unittest.TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.se2 = RBF(1, active_dims=[2])
        self.rq0 = RationalQuadratic(1, active_dims=[0])
        self.rq1 = RationalQuadratic(1, active_dims=[1])
        self.rq2 = RationalQuadratic(1, active_dims=[2])
        self.lin0 = LinScaleShift(1, active_dims=[0])

    def test_create_grammar(self):
        base_kernel_names = ['SE', 'RQ']
        dim = 1

        # Test with base kernel names and dimension arguments
        grammar = CKSGrammar(base_kernel_names=base_kernel_names, n_dims=dim)

        self.assertEqual(base_kernel_names, grammar.base_kernel_names)
        self.assertEqual(len(base_kernel_names), len(grammar.base_kernel_names))
        self.assertEqual(dim, grammar.n_dims)
        self.assertEqual(None, grammar.hyperpriors)
        self.assertTrue(base_kernel_eq(self.se0, grammar.base_kernels[0]))
        self.assertTrue(base_kernel_eq(self.rq0, grammar.base_kernels[1]))
        self.assertFalse(base_kernel_eq(self.se0, grammar.base_kernels[1]))

    def test_mask_kernels_multi_d(self):
        base_kernel_names = ['SE', 'RQ']
        dim = 3

        # Test with base kernel names and dimension arguments
        grammar = CKSGrammar(base_kernel_names=base_kernel_names, n_dims=dim)

        expected_kernels = [
            self.se0,
            self.se1,
            self.se2,
            self.rq0,
            self.rq1,
            self.rq2,
        ]

        for expected_kernel, actual_kernel in zip(expected_kernels, grammar.base_kernels):
            self.assertTrue(base_kernel_eq(expected_kernel, actual_kernel))

    def test_initialize(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        result = grammar.initialize()
        self.assertIsInstance(result, list)

        kernel_types = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertEqual(len(result), len(kernel_types))
        kernels = [k.kernel for k in result]
        k_types_exist = [has_combo_kernel_type(kernels, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        scored = [k.evaluated for k in result]
        self.assertFalse(all(scored))
        nan_scored = [k.nan_scored for k in result]
        self.assertFalse(all(nan_scored))

        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=1)
        result = grammar.initialize()
        self.assertIsInstance(result, list)

        kernel_types = [self.se0, self.rq0]
        self.assertEqual(len(result), len(kernel_types))
        kernels = [k.kernel for k in result]
        k_types_exist = [has_combo_kernel_type(kernels, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        scored = [k.evaluated for k in result]
        self.assertFalse(all(scored))
        nan_scored = [k.nan_scored for k in result]
        self.assertFalse(all(nan_scored))

    def test_expand(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        scored_kernel = GPModel(self.se0)
        scored_kernel.score = 1
        result = grammar.expand([scored_kernel])
        self.assertIsInstance(result, list)
        # TODO: test that expand_full_kernel is called with each kernel

    def test_expand_one_dim(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ', 'LIN'], n_dims=1)

        # Expand SE
        se = grammar.base_kernels[0]
        expected_kernels = [
            self.se0 + self.se0,
            self.se0 * self.se0,
            self.se0 + self.rq0,
            self.se0 * self.rq0,
            self.se0 + self.lin0,
            self.se0 * self.lin0,
        ]
        new_kernels = grammar.expand_single_kernel(se)
        k_types_exist = [has_combo_kernel_type(new_kernels, k_type) for k_type in expected_kernels]
        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(expected_kernels), len(new_kernels))

        # Expand SE + RQ
        se_plus_rq = self.se0 + self.rq0
        expected_kernels = [
            (self.se0 + self.rq0) + self.se0,
            (self.se0 + self.rq0) * self.se0,
            (self.se0 + self.rq0) + self.rq0,
            (self.se0 + self.rq0) * self.rq0,
            (self.se0 + self.rq0) + self.lin0,
            (self.se0 + self.rq0) * self.lin0,
        ]
        new_kernels = grammar.expand_single_kernel(se_plus_rq)
        k_types_exist = [has_combo_kernel_type(new_kernels, k_type) for k_type in expected_kernels]
        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(expected_kernels), len(new_kernels))

    def test_expand_single_kernel_two_dims(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        # first, test 1d expansion of base kernel
        k = self.se0
        result = grammar.expand_single_kernel(k)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = grammar.expand_single_kernel(k)

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

    def test_expand_single_kernel_mutli_d(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=3)

        # Expand SE
        se = grammar.base_kernels[0]
        expected_kernels = [
            self.se0 + self.se0,
            self.se0 * self.se0,
            self.se0 + self.se1,
            self.se0 * self.se1,
            self.se0 + self.se2,
            self.se0 * self.se2,
            self.se0 + self.rq0,
            self.se0 * self.rq0,
            self.se0 + self.rq1,
            self.se0 * self.rq1,
            self.se0 + self.rq2,
            self.se0 * self.rq2,
        ]
        new_kernels = grammar.expand_single_kernel(se)
        k_types_exist = [has_combo_kernel_type(new_kernels, k_type) for k_type in expected_kernels]
        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(expected_kernels), len(new_kernels))

    def test_expand_full_brute_force_level_0(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        n = len(grammar.base_kernel_names)
        n_dim = grammar.n_dims
        max_number_of_models = 1000

        level = 0
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = n * n_dim
        self.assertEqual(expected, len(kernels))

    def test_expand_full_brute_force_level_1(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        n = len(grammar.base_kernel_names)
        n_dim = grammar.n_dims
        max_number_of_models = 1000

        level = 1
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = (n * n_dim + 1) * n * n_dim
        self.assertEqual(expected, len(kernels))

    @unittest.skip("Skipping expand full brute force test level 2 and 3 in the interest of time.")
    def test_expand_full_brute_force_level_2_and_3(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        max_number_of_models = 1000

        level = 2
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = 134
        self.assertEqual(expected, len(kernels))

        level = 3
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = 834  # Question: should this be = max_number_of_models ?
        self.assertEqual(expected, len(kernels))

    def test_expand_full_kernel(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        # first, test 1d expansion of base kernel
        k = self.se0
        result = grammar.expand_full_kernel(k)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = grammar.expand_full_kernel(k)

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1,
                        (self.se1 + self.se0) * self.rq1, (self.se1 + self.rq0) * self.rq1,
                        (self.se1 + self.se1) * self.rq1, (self.se1 + self.rq1) * self.rq1,
                        (self.se1 * self.se0) * self.rq1, (self.se1 * self.rq0) * self.rq1,
                        (self.se1 * self.se1) * self.rq1, (self.se1 * self.rq1) * self.rq1,

                        self.se1 * (self.rq1 + self.se0), self.se1 * (self.rq1 + self.rq0),
                        self.se1 * (self.rq1 + self.se1), self.se1 * (self.rq1 + self.rq1),
                        self.se1 * (self.rq1 * self.se0), self.se1 * (self.rq1 * self.rq0),
                        self.se1 * (self.rq1 * self.se1), self.se1 * (self.rq1 * self.rq1),
                        ]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))


class TestBomsGrammar(unittest.TestCase):

    def test_create(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        # No optional arguments
        grammar = BOMSGrammar(base_kernel_names=base_kernel_names, n_dims=n_dim)
        self.assertEqual(base_kernel_names, grammar.base_kernel_names)
        self.assertEqual(n_dim, grammar.n_dims)
        self.assertIsNotNone(grammar.hyperpriors)

        self.assertIsNotNone(grammar.random_walk_geometric_dist_parameter)
        self.assertIsNotNone(grammar.number_of_top_k_best)
        self.assertIsNotNone(grammar.number_of_random_walks)

    def test_get_candidates_empty(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        # No optional arguments
        grammar = BOMSGrammar(base_kernel_names=base_kernel_names, n_dims=n_dim)

        seed = np.random.randint(100)
        np.random.seed(seed)
        candidates = grammar.get_candidates([])

        np.random.seed(seed)
        total_num_walks = grammar.number_of_random_walks
        expected_candidates = grammar.expand_random(total_num_walks)

        self.assertEqual(len(expected_candidates), len(candidates))

    def test_get_candidates(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        grammar = BOMSGrammar(base_kernel_names=base_kernel_names, n_dims=n_dim)

        grammar.number_of_top_k_best = 1
        grammar.num_random_walks = 5
        kernels = grammar.expand_random(grammar.number_of_random_walks)
        fitness_score = np.random.permutation(len(kernels))

        models = [GPModel(kernel) for kernel in kernels]
        for model, model_score in zip(models, fitness_score):
            model.score = model_score

        candidates = grammar.get_candidates(models)
        for candidate in candidates:
            self.assertIsInstance(candidate, GPModel)

    def test_expand(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2
        num_random_walks = 5

        grammar = BOMSGrammar(base_kernel_names=base_kernel_names, n_dims=n_dim)
        grammar.random_walk_geometric_dist_parameter = 1 / 3
        grammar.number_of_random_walks = 1

        np.random.seed(5)
        new_kernels = grammar.expand_random(num_random_walks)
        self.assertEqual(len(new_kernels), num_random_walks)
        self.assertIsInstance(new_kernels[0], Kern)
        self.assertNotIsInstance(new_kernels[0], CombinationKernel)

    def test_expand_best(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 1

        np.random.seed(5)
        grammar = BOMSGrammar(base_kernel_names=base_kernel_names, n_dims=n_dim)

        grammar.number_of_top_k_best = 1
        num_random_walks = 5
        kernels = grammar.expand_random(num_random_walks)
        fitness_score = list(np.random.permutation(len(kernels)).tolist())

        index = int(np.argmax(fitness_score))
        kernel_to_expand = GPModel(kernels[index])

        models = [GPModel(kernel) for kernel in kernels]
        for model, model_score in zip(models, fitness_score):
            model.score = model_score

        new_kernels = grammar.expand_best(models, fitness_score)

        expanded_kernels = grammar.expand([kernel_to_expand])

        for i in range(len(expanded_kernels)):
            self.assertTrue(has_combo_kernel_type([new_kernels[i]], expanded_kernels[i].kernel))

    def tearDown(self) -> None:
        np.random.seed()

import unittest

import numpy as np
from GPy.kern import RBF, RationalQuadratic, LinScaleShift
from GPy.kern.src.kern import CombinationKernel

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import BaseGrammar, CKSGrammar, BomsGrammar


class TestBaseGrammar(unittest.TestCase):

    def setUp(self):
        self.grammar = BaseGrammar(['SE', 'RQ'])
        self.grammar.build(1)

    def test_expand(self):
        seed_kernel = GPModel(RBF(1))
        self.assertRaises(NotImplementedError, self.grammar.expand, seed_kernel)


class TestCKSGrammar(unittest.TestCase):

    def setUp(self):
        self.se0 = Covariance(RBF(1, active_dims=[0]))
        self.se1 = Covariance(RBF(1, active_dims=[1]))
        self.se2 = Covariance(RBF(1, active_dims=[2]))
        self.rq0 = Covariance(RationalQuadratic(1, active_dims=[0]))
        self.rq1 = Covariance(RationalQuadratic(1, active_dims=[1]))
        self.rq2 = Covariance(RationalQuadratic(1, active_dims=[2]))
        self.lin0 = Covariance(LinScaleShift(1, active_dims=[0]))

    def test_create_grammar(self):
        base_kernel_names = ['SE', 'RQ']
        dim = 1

        # Test with base kernel names and dimension arguments
        grammar = CKSGrammar(base_kernel_names=base_kernel_names)
        grammar.build(dim)

        self.assertEqual(base_kernel_names, grammar.base_kernel_names)
        self.assertEqual(len(base_kernel_names), len(grammar.base_kernel_names))
        self.assertEqual(dim, grammar.n_dims)
        self.assertEqual(None, grammar.hyperpriors)
        self.assertEqual(self.se0.infix, grammar.base_kernels[0].infix)
        self.assertEqual(self.rq0.infix, grammar.base_kernels[1].infix)
        self.assertNotEqual(self.se0.infix, grammar.base_kernels[1].infix)

    def test_mask_kernels_multi_d(self):
        base_kernel_names = ['SE', 'RQ']
        dim = 3

        # Test with base kernel names and dimension arguments
        grammar = CKSGrammar(base_kernel_names=base_kernel_names)
        grammar.build(dim)

        expected_kernels = [
            self.se0,
            self.se1,
            self.se2,
            self.rq0,
            self.rq1,
            self.rq2,
        ]

        for expected_kernel, actual_kernel in zip(expected_kernels, grammar.base_kernels):
            self.assertEqual(expected_kernel.infix, actual_kernel.infix)

    def test_expand(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)
        scored_kernel = GPModel(self.se0)
        scored_kernel.score = 1
        result = grammar.expand([scored_kernel])
        self.assertIsInstance(result, list)
        # TODO: test that expand_full_kernel is called with each kernel

    def test_expand_one_dim(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ', 'LIN'])
        grammar.build(n_dims=1)

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
        self.assertIsInstance(new_kernels, list)
        self.assertEqual(len(expected_kernels), len(new_kernels))
        for expected_cov, actual_cov in zip(expected_kernels, new_kernels):
            self.assertEqual(expected_cov.infix, actual_cov.infix)

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
        self.assertIsInstance(new_kernels, list)
        self.assertEqual(len(expected_kernels), len(new_kernels))
        for expected_cov, actual_cov in zip(expected_kernels, new_kernels):
            self.assertEqual(expected_cov.infix, actual_cov.infix)

    def test_expand_single_kernel_two_dims(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)

        # first, test 1d expansion of base kernel
        k = self.se0
        expected_kernels = grammar.expand_single_kernel(k)

        new_kernels = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                       self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1]
        self.assertIsInstance(new_kernels, list)
        self.assertEqual(len(expected_kernels), len(new_kernels))
        new_kernels_infixes = [k.infix for k in new_kernels]
        expected_infixes = [k.infix for k in expected_kernels]
        self.assertCountEqual(expected_infixes, new_kernels_infixes)

        # test combination kernel expansion
        k = self.se1 * self.rq1
        new_kernels = grammar.expand_single_kernel(k)

        expected_kernels = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                            self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                            self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                            self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1]
        new_kernels_infixes = [k.infix for k in new_kernels]
        expected_infixes = [k.infix for k in expected_kernels]
        self.assertCountEqual(expected_infixes, new_kernels_infixes)

    def test_expand_single_kernel_mutli_d(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=3)

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
        self.assertIsInstance(new_kernels, list)
        self.assertEqual(len(expected_kernels), len(new_kernels))
        for expected_cov, actual_cov in zip(expected_kernels, new_kernels):
            self.assertEqual(expected_cov.infix, actual_cov.infix)

    def test_expand_full_brute_force_level_0(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)

        n = len(grammar.base_kernel_names)
        n_dim = grammar.n_dims
        max_number_of_models = 1000

        level = 0
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = n * n_dim
        self.assertEqual(expected, len(kernels))

    def test_expand_full_brute_force_level_1(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)
        n = len(grammar.base_kernel_names)
        n_dim = grammar.n_dims
        max_number_of_models = 1000

        level = 1
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = (n * n_dim + 1) * n * n_dim
        self.assertEqual(expected, len(kernels))

    @unittest.skip("Skipping expand full brute force test level 2 and 3 in the interest of time.")
    def test_expand_full_brute_force_level_2_and_3(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)
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
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'])
        grammar.build(n_dims=2)

        # first, test 1d expansion of base kernel
        k = self.se0
        new_kernels = grammar.expand_full_kernel(k)

        expected_kernels = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                            self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1]
        new_kernels_infixes = [k.infix for k in new_kernels]
        expected_infixes = [k.infix for k in expected_kernels]
        self.assertCountEqual(expected_infixes, new_kernels_infixes)

        # test combination kernel expansion
        k = self.se1 * self.rq1
        new_kernels = grammar.expand_full_kernel(k)

        expected_kernels = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
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
        new_kernels_infixes = [k.infix for k in new_kernels]
        expected_infixes = [k.infix for k in expected_kernels]
        self.assertCountEqual(expected_infixes, new_kernels_infixes)


class TestBomsGrammar(unittest.TestCase):

    def test_create(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        # No optional arguments
        grammar = BomsGrammar(base_kernel_names=base_kernel_names)
        grammar.build(n_dim)

        self.assertEqual(base_kernel_names, grammar.base_kernel_names)
        self.assertEqual(n_dim, grammar.n_dims)
        self.assertIsNotNone(grammar.hyperpriors)

        self.assertIsNotNone(grammar._random_walk_geometric_dist_parameter)
        self.assertIsNotNone(grammar._number_of_top_k_best)
        self.assertIsNotNone(grammar._number_of_random_walks)

    def test_get_candidates_empty(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        # No optional arguments
        grammar = BomsGrammar(base_kernel_names=base_kernel_names)
        grammar.build(n_dim)

        seed = np.random.randint(100)
        np.random.seed(seed)
        candidates = grammar.get_candidates([])

        np.random.seed(seed)
        total_num_walks = grammar._number_of_random_walks
        expected_candidates = grammar.expand_random(total_num_walks)

        self.assertEqual(len(expected_candidates), len(candidates))

    def test_get_candidates(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2

        grammar = BomsGrammar(base_kernel_names=base_kernel_names)
        grammar.build(n_dim)

        grammar._number_of_top_k_best = 1
        grammar.num_random_walks = 5
        kernels = grammar.expand_random(grammar._number_of_random_walks)
        fitness_score = np.random.permutation(len(kernels))

        models = [GPModel(kernel) for kernel in kernels]

        candidates = grammar.get_candidates(models)
        for candidate in candidates:
            self.assertIsInstance(candidate, Covariance)

    def test_expand(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 2
        num_random_walks = 5

        grammar = BomsGrammar(base_kernel_names=base_kernel_names)
        grammar.build(n_dim)
        grammar._random_walk_geometric_dist_parameter = 1 / 3
        grammar._number_of_random_walks = 1

        np.random.seed(5)
        new_kernels = grammar.expand_random(num_random_walks)
        self.assertEqual(len(new_kernels), num_random_walks)
        self.assertIsInstance(new_kernels[0], Covariance)
        self.assertNotIsInstance(new_kernels[0], CombinationKernel)

    def test_expand_best(self):
        base_kernel_names = ['SE', 'RQ']
        n_dim = 1

        np.random.seed(5)
        grammar = BomsGrammar(base_kernel_names=base_kernel_names)
        grammar.build(n_dim)

        grammar._number_of_top_k_best = 1
        num_random_walks = 5
        kernels = grammar.expand_random(num_random_walks)
        fitness_score = list(np.random.permutation(len(kernels)).tolist())

        index = int(np.argmax(fitness_score))
        kernel_to_expand = kernels[index]

        models = [GPModel(kernel) for kernel in kernels]
        for model, model_score in zip(models, fitness_score):
            model.score = model_score

        new_kernels = grammar.expand_best(models, fitness_score)

        expanded_kernels = grammar.expand_single_kernel(kernel_to_expand)

        for i in range(len(expanded_kernels)):
            self.assertEqual(new_kernels[i].infix, expanded_kernels[i].infix)

    def tearDown(self) -> None:
        np.random.seed()

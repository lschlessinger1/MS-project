from unittest import TestCase

from GPy.kern import RBF, RationalQuadratic, Add, LinScaleShift

from src.autoks.grammar import BaseGrammar, CKSGrammar, remove_duplicate_kernels
from src.autoks.kernel import AKSKernel
from src.test.autoks.support.util import has_combo_kernel_type, base_kernel_eq


class TestGrammar(TestCase):

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


class TestBaseGrammar(TestCase):

    def setUp(self):
        self.grammar = BaseGrammar(['SE', 'RQ'], 1)

    def test_initialize(self):
        self.assertRaises(NotImplementedError, self.grammar.initialize)

    def test_expand(self):
        seed_kernel = AKSKernel(RBF(1))
        self.assertRaises(NotImplementedError, self.grammar.expand, seed_kernel)


class TestCKSGrammar(TestCase):

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
        scored_kernel = AKSKernel(self.se0)
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

    def test_expand_full_brute_force(self):
        grammar = CKSGrammar(base_kernel_names=['SE', 'RQ'], n_dims=2)
        n = len(grammar.base_kernel_names)
        n_dim = grammar.n_dims
        max_number_of_models = 1000

        level = 0
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = n * n_dim
        self.assertEqual(expected, len(kernels))

        level = 1
        kernels = grammar.expand_full_brute_force(level, max_number_of_models)
        expected = (n * n_dim + 1) * n * n_dim
        self.assertEqual(expected, len(kernels))

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

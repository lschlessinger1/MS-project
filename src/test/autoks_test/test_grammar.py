from unittest import TestCase

from GPy.kern import RBF, RatQuad, Add

from src.autoks.grammar import BaseGrammar, CKSGrammar, remove_duplicate_kernels
from src.autoks.kernel import AKSKernel
from src.test.autoks_test.support.util import has_combo_kernel_type


class TestGrammar(TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

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
        self.grammar = BaseGrammar()

    def test_initialize(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.initialize(['SE', 'RQ'], 1)

    def test_expand(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.expand('', '', '', '')


class TestCKSGrammar(TestCase):

    def setUp(self):
        self.operators = ['+', '*']
        self.grammar = CKSGrammar()
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

    def test_initialize(self):
        result = self.grammar.initialize(['SE', 'RQ'], n_dims=2)
        self.assertIsInstance(result, list)

        kernel_types = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertEqual(len(result), len(kernel_types))
        kernels = [k.kernel for k in result]
        k_types_exist = [has_combo_kernel_type(kernels, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        scored = [k.scored for k in result]
        self.assertFalse(all(scored))
        nan_scored = [k.nan_scored for k in result]
        self.assertFalse(all(nan_scored))

        result = self.grammar.initialize(['SE', 'RQ'], n_dims=1)
        self.assertIsInstance(result, list)

        kernel_types = [self.se0, self.rq0]
        self.assertEqual(len(result), len(kernel_types))
        kernels = [k.kernel for k in result]
        k_types_exist = [has_combo_kernel_type(kernels, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        scored = [k.scored for k in result]
        self.assertFalse(all(scored))
        nan_scored = [k.nan_scored for k in result]
        self.assertFalse(all(nan_scored))

    def test_expand(self):
        scored_kernel = AKSKernel(self.se0)
        scored_kernel.score = 1
        result = self.grammar.expand([scored_kernel], ['SE', 'RQ'], 2)
        self.assertIsInstance(result, list)
        # TODO: test that expand_full_kernel is called with each kernel

    def test_expand_single_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_single_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_single_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

    def test_expand_full_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_full_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_full_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1,
                        (self.se1 + self.se0) * self.rq1, (self.se1 + self.rq0) * self.rq1,
                        (self.se1 + self.se1) * self.rq1, (self.se1 + self.rq1) * self.rq1,
                        (self.se1 * self.se0) * self.rq1, (self.se1 * self.rq0) * self.rq1,
                        (self.se1 * self.se1) * self.rq1, (self.se1 * self.rq1) * self.rq1,
                        self.se1 * self.se1, self.rq1 * self.rq1,
                        self.se1 * (self.rq1 + self.se0), self.se1 * (self.rq1 + self.rq0),
                        self.se1 * (self.rq1 + self.se1), self.se1 * (self.rq1 + self.rq1),
                        self.se1 * (self.rq1 * self.se0), self.se1 * (self.rq1 * self.rq0),
                        self.se1 * (self.rq1 * self.se1), self.se1 * (self.rq1 * self.rq1),
                        self.se1 * self.se1, self.se1 * self.rq1]
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]

        self.assertTrue(all(k_types_exist))
        self.assertEqual(len(result), len(kernel_types))

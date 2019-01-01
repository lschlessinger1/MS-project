from collections import Counter
from unittest import TestCase

import numpy as np
from GPy.kern import RBF, RatQuad, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern

from autoks.grammar import BaseGrammar, CKSGrammar


class TestBaseGrammar(TestCase):

    def setUp(self):
        self.k = 4
        self.grammar = BaseGrammar(self.k)

    def test_initialize(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.initialize('', '', '')

    def test_expand(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.expand('', '', '')

    def test_select(self):
        result = self.grammar.select(np.array([1, 2, 3, 4, 5]), np.array([.1, .2, .3, .4, .5]))
        self.assertEqual(len(result), self.k)


class TestCKSGrammar(TestCase):

    def setUp(self):
        self.k = 4
        self.grammar = CKSGrammar(self.k)
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

    # Helpers #
    @staticmethod
    def counter_repr(x):
        return Counter(frozenset(Counter(item).items()) for item in x)

    @staticmethod
    def lists_equal_without_order(a, b):
        return TestCKSGrammar.counter_repr(a) == TestCKSGrammar.counter_repr(b)

    @staticmethod
    def same_combo_type(k1, k2):
        return isinstance(k1, Add) and isinstance(k2, Add) or isinstance(k1, Prod) and isinstance(k2, Prod)

    def has_combo_kernel_type(self, kernels, kern):
        is_combo_kernel = isinstance(kern, CombinationKernel)
        is_base_kernel = isinstance(kern, Kern) and not is_combo_kernel
        for kernel in kernels:
            if isinstance(kernel, CombinationKernel) and is_combo_kernel:
                kparts = [(k.__class__, k.active_dims[0]) for k in kern.parts]
                part_list = [(part.__class__, part.active_dims[0]) for part in kernel.parts]
                same_combo = self.same_combo_type(kernel, kern)
                if self.lists_equal_without_order(kparts, part_list) and same_combo:
                    return True
            elif isinstance(kernel, Kern) and is_base_kernel:
                same_type = kernel.name == kern.name
                same_dim = len(kernel.active_dims) == 1 and kernel.active_dims[0] == kern.active_dims[0]
                if same_type and same_dim:
                    return True
        return False

    def test_expand_single_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_single_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1]
        ktypes_exist = [self.has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_single_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1]
        ktypes_exist = [self.has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

    def test_expand_full_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_full_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        ktypes_exist = [self.has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_full_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

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
        ktypes_exist = [self.has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

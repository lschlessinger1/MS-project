from collections import Counter
from unittest import TestCase

import numpy as np
from GPy.kern import RBF, RatQuad, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern

from src.autoks.grammar import BaseGrammar, CKSGrammar, remove_duplicate_kernels
from src.autoks.kernel import AKSKernel


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
        self.k = 4
        self.max_candidates = 100
        self.max_offspring = 1000
        self.grammar = BaseGrammar(self.k, self.max_candidates, self.max_offspring)

    def test_initialize(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.initialize(['SE', 'RQ'], 10, 1)

    def test_expand(self):
        with self.assertRaises(NotImplementedError):
            self.grammar.expand('', '', '', '')

    def test_select_parents(self):
        se0 = RBF(1, active_dims=[0])
        kernels = [AKSKernel(se0) for i in range(5)]
        for i, k in enumerate(kernels):
            k.score = i
        result = self.grammar.select_parents(np.array(kernels))
        self.assertEqual(len(result), self.k)


class TestCKSGrammar(TestCase):

    def setUp(self):
        self.k = 4
        self.operators = ['+', '*']
        self.grammar = CKSGrammar(self.k, 100, 1000)
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

    def test_expand_single_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_single_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_single_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se1 * self.rq1 + self.se0, self.se1 * self.rq1 + self.rq0,
                        self.se1 * self.rq1 + self.se1, self.se1 * self.rq1 + self.rq1,
                        self.se1 * self.rq1 * self.se0, self.se1 * self.rq1 * self.rq0,
                        self.se1 * self.rq1 * self.se1, self.se1 * self.rq1 * self.rq1]
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

    def test_expand_full_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_full_kernel(k, 2, ['SE', 'RQ'], self.operators)

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
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
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))


# Helpers #
def counter_repr(x):
    return Counter(frozenset(Counter(item).items()) for item in x)


def lists_equal_without_order(a, b):
    return counter_repr(a) == counter_repr(b)


def same_combo_type(k1, k2):
    return isinstance(k1, Add) and isinstance(k2, Add) or isinstance(k1, Prod) and isinstance(k2, Prod)


def has_combo_kernel_type(kernels, kern):
    is_combo_kernel = isinstance(kern, CombinationKernel)
    is_base_kernel = isinstance(kern, Kern) and not is_combo_kernel
    for kernel in kernels:
        if isinstance(kernel, CombinationKernel) and is_combo_kernel:
            kparts = [(k.__class__, k.active_dims[0]) for k in kern.parts]
            part_list = [(part.__class__, part.active_dims[0]) for part in kernel.parts]
            same_combo = same_combo_type(kernel, kern)
            if lists_equal_without_order(kparts, part_list) and same_combo:
                return True
        elif isinstance(kernel, Kern) and is_base_kernel:
            same_type = kernel.name == kern.name
            same_dim = len(kernel.active_dims) == 1 and kernel.active_dims[0] == kern.active_dims[0]
            if same_type and same_dim:
                return True
    return False

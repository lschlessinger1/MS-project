from collections import Counter
from unittest import TestCase

import numpy as np
from GPy.kern import RBF, RatQuad, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern
from GPy.models import GPRegression

from autoks.grammar import BaseGrammar, CKSGrammar, sort_kernel, remove_duplicates, remove_duplicate_models


class TestGrammar(TestCase):

    def setUp(self):
        self.se0 = RBF(1, active_dims=[0])
        self.se1 = RBF(1, active_dims=[1])
        self.rq0 = RatQuad(1, active_dims=[0])
        self.rq1 = RatQuad(1, active_dims=[1])

    def test_sort_kernel(self):
        k = self.se0
        result = sort_kernel(k)
        # should be SE0
        self.assertEqual(result.active_dims[0], 0)
        self.assertIsInstance(result, RBF)

        k = self.se1 + self.se0
        result = sort_kernel(k)
        # should be SE0 + SE1
        self.assertIsInstance(result, Add)
        self.assertIsInstance(result.parts[0], RBF)
        self.assertIsInstance(result.parts[1], RBF)
        self.assertEqual(result.parts[0].active_dims[0], 0)
        self.assertEqual(result.parts[1].active_dims[0], 1)

        k = self.se1 + self.se0 + self.rq1 + self.rq0
        result = sort_kernel(k)
        # should be RQ0 + RQ1 + SE0 + SE1
        self.assertIsInstance(result, Add)
        kernel_types = [(RatQuad, 0), (RatQuad, 1), (RBF, 0), (RBF, 1)]
        for (k_class, dim), part in zip(kernel_types, result.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        k = self.rq1 * self.rq0 + self.se1 * self.se0 * self.rq1 + self.se1 * self.rq0 * self.rq1
        result = sort_kernel(k)
        # should be (RQ1 * SE0 * SE1) + (RQ0 * RQ1) + (RQ0 * RQ1 * SE1)
        self.assertIsInstance(result, Add)
        kernel_types_outer = [Prod, Prod, Prod]
        for k_class, part in zip(kernel_types_outer, result.parts):
            self.assertIsInstance(part, k_class)

        kernel_types_inner_1 = [(RatQuad, 1), (RBF, 0), (RBF, 1)]
        prod_1 = result.parts[0]
        for (k_class, dim), part in zip(kernel_types_inner_1, prod_1.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_2 = [(RatQuad, 0), (RatQuad, 1)]
        prod_2 = result.parts[1]
        for (k_class, dim), part in zip(kernel_types_inner_2, prod_2.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

        kernel_types_inner_3 = [(RatQuad, 0), (RatQuad, 1), (RBF, 1)]
        prod_3 = result.parts[2]
        for (k_class, dim), part in zip(kernel_types_inner_3, prod_3.parts):
            self.assertIsInstance(part, k_class)
            self.assertEqual(part.active_dims[0], dim)

    def test_remove_duplicates(self):
        # simple example
        data = [1, 1, 1, 2, 3, 4, 'a', 'b', True]
        values = [10, 9, 8, '7', '6', False, 4, 3, 2]
        result = remove_duplicates(data, values)
        self.assertEqual(result, [10, '7', '6', False, 4, 3])

        with self.assertRaises(ValueError):
            remove_duplicates([1, 2, 3], ['1', 2])

    def test_remove_duplicate_models(self):
        x = np.array([[1, 2], [4, 5]])
        y = np.zeros((x.shape[0], 1))

        kernels = [self.se0 + self.se0, self.se1, self.se0, self.se0, self.se1 + self.se0, self.se0 + self.se1]
        models = []
        for kern in kernels:
            models.append(GPRegression(X=x, Y=y, kernel=kern))

        models_pruned = remove_duplicate_models(models)
        kernels_pruned = [m.kern for m in models_pruned]
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

    def test_expand_single_kernel(self):
        # first, test 1d expansion of base kernel
        k = self.se0
        result = self.grammar.expand_single_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

        self.assertTrue(all(ktypes_exist))
        self.assertEqual(len(result), len(kernel_types))

        # test combination kernel expansion
        k = self.se1 * self.rq1
        result = self.grammar.expand_single_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

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
        result = self.grammar.expand_full_kernel(k, ['+', '*'], 2, ['SE', 'RQ'])

        kernel_types = [self.se0 + self.se0, self.se0 + self.rq0, self.se0 + self.se1, self.se0 + self.rq1,
                        self.se0 * self.se0, self.se0 * self.rq0, self.se0 * self.se1, self.se0 * self.rq1,
                        self.se0, self.rq0]
        ktypes_exist = [has_combo_kernel_type(result, ktype) for ktype in kernel_types]

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

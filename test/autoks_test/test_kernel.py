import unittest

from GPy.kern import RBF, Add, RatQuad, Prod

from autoks.kernel import sort_kernel
from autoks.util import remove_duplicates


class TestKernel(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
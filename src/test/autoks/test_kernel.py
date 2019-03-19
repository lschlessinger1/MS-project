import unittest

from GPy.kern import RBF, Add, RatQuad, Prod

from src.autoks.kernel import sort_kernel, AKSKernel, get_all_1d_kernels, create_1d_kernel
from src.autoks.util import remove_duplicates
from src.evalg.encoding import BinaryTree
from src.test.autoks.support.util import has_combo_kernel_type


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

    def test_get_all_1d_kernels(self):
        result = get_all_1d_kernels(['SE', 'RQ'], n_dims=2)
        self.assertIsInstance(result, list)
        kernel_types = [self.se0, self.se1, self.rq0, self.rq1]
        self.assertEqual(len(result), len(kernel_types))
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

        result = get_all_1d_kernels(['SE', 'RQ'], n_dims=1)
        self.assertIsInstance(result, list)
        kernel_types = [self.se0, self.rq0]
        self.assertEqual(len(result), len(kernel_types))
        k_types_exist = [has_combo_kernel_type(result, k_type) for k_type in kernel_types]
        self.assertTrue(all(k_types_exist))

    def test_create_1d_kernel(self):
        result = create_1d_kernel(active_dim=0, kernel_family='SE')
        self.assertIsInstance(result, RBF)
        self.assertEqual(result.active_dims[0], 0)

        result = create_1d_kernel(active_dim=1, kernel_family='RQ')
        self.assertIsInstance(result, RatQuad)
        self.assertEqual(result.active_dims[0], 1)


class TestAKSKernel(unittest.TestCase):

    def test_to_binary_tree(self):
        kernel = RBF(1) * RBF(1) + RatQuad(1)
        aks_kernel = AKSKernel(kernel)
        result = aks_kernel.to_binary_tree()
        self.assertIsInstance(result, BinaryTree)
        self.assertCountEqual(result.postfix_tokens(), ['SE0', 'SE0', '*', 'RQ0', '+'])


if __name__ == '__main__':
    unittest.main()

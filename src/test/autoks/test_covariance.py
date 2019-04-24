from unittest import TestCase

from graphviz import Source

from src.autoks.backend.kernel import get_all_1d_kernels, RawKernelType
from src.autoks.core.covariance import Covariance
from src.autoks.core.kernel_encoding import KernelTree
from src.autoks.symbolic.kernel_symbol import KernelSymbol


class TestCovariance(TestCase):

    def setUp(self) -> None:
        base_kernels = get_all_1d_kernels(['SE', 'RQ'], 2)
        self.se_0 = base_kernels[0]
        self.se_1 = base_kernels[1]
        self.rq_0 = base_kernels[2]
        self.rq_1 = base_kernels[3]

    def test_create_empty(self):
        self.assertRaises(TypeError, Covariance)

    def test_create_one_d(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertEqual(kern, cov.raw_kernel)
        self.assertListEqual([kern], cov.infix_tokens)
        self.assertIsInstance(cov.infix, str)
        self.assertIsInstance(cov.infix_full, str)
        self.assertEqual('SE_0', cov.infix)
        self.assertEqual('SE_0', cov.postfix)
        self.assertListEqual([kern], cov.postfix_tokens)
        self.assertIsInstance(cov.symbolic_expr, KernelSymbol)
        self.assertEqual(kern, cov.symbolic_expr.kernel_one_d)
        self.assertIsInstance(cov.symbolic_expr_expanded, KernelSymbol)
        self.assertEqual(kern, cov.symbolic_expr_expanded.kernel_one_d)

    def test_to_binary_tree(self):
        kern = self.se_0
        cov = Covariance(kern)
        tree = cov.to_binary_tree()
        self.assertIsInstance(tree, KernelTree)

    def test_canonical(self):
        kern = self.se_0
        cov = Covariance(kern)
        kern = cov.canonical()
        self.assertIsInstance(kern, RawKernelType)

    def test_to_additive_form(self):
        kern = self.se_0
        cov = Covariance(kern)
        kern = cov.to_additive_form()
        self.assertIsInstance(kern, RawKernelType)

    def test_symbolically_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolically_equals(cov_2))
        self.assertTrue(cov_2.symbolically_equals(cov_1))

        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolically_equals(cov_2))
        self.assertFalse(cov_2.symbolically_equals(cov_1))

    def test_symbolic_expanded_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolic_expanded_equals(cov_2))
        self.assertFalse(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.symbolic_expanded_equals(cov_2))
        self.assertFalse(cov_2.symbolic_expanded_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

        # Test additive equivalence
        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.symbolic_expanded_equals(cov_2))
        self.assertTrue(cov_2.symbolic_expanded_equals(cov_1))

    def test_infix_equals(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.infix_equals(cov_2))
        self.assertTrue(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0
        kern_2 = self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_0 * self.se_1
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertTrue(cov_1.infix_equals(cov_2))
        self.assertTrue(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 + self.se_1
        kern_2 = self.se_1 + self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.rq_0 + self.se_0 * self.se_1
        kern_2 = self.se_1 * self.se_0 + self.rq_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * (self.se_1 + self.rq_0)
        kern_2 = self.se_1 * (self.se_0 + self.rq_0)
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        kern_1 = self.se_0 * self.se_1 + self.se_1 * self.rq_0
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

        # Test additive equivalence
        kern_1 = (self.se_0 + self.rq_0) * self.se_1
        kern_2 = self.se_1 * self.rq_0 + self.se_1 * self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)
        self.assertFalse(cov_1.infix_equals(cov_2))
        self.assertFalse(cov_2.infix_equals(cov_1))

    def test_is_base(self):
        kern = self.se_0
        cov = Covariance(kern)
        self.assertTrue(cov.is_base())

        kern = self.se_0 + self.se_0
        cov = Covariance(kern)
        self.assertFalse(cov.is_base())

    def test_add(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)

        cov = cov_1 + cov_2
        self.assertEqual('SE_0 + SE_0', cov.infix)

    def test_multiply(self):
        kern_1 = self.se_0
        kern_2 = self.se_0
        cov_1 = Covariance(kern_1)
        cov_2 = Covariance(kern_2)

        cov = cov_1 * cov_2
        self.assertEqual('SE_0 * SE_0', cov.infix)

    def test_as_latex(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_latex()
        self.assertIsInstance(actual, str)

    def test_as_mathml(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_mathml()
        self.assertIsInstance(actual, str)

    def test_as_dot(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_dot()
        self.assertIsInstance(actual, str)

    def test_as_graph(self):
        kern = self.se_0
        cov = Covariance(kern)
        actual = cov.as_graph()
        self.assertIsInstance(actual, Source)

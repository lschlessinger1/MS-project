from unittest import TestCase

from GPy.kern import RBF

from src.autoks.symbolic.kernel_symbol import KernelSymbol


class TestKernelSymbol(TestCase):

    def test_init(self):
        se_0 = RBF(1)
        name = 'SE_0'
        actual = KernelSymbol(name, se_0)
        self.assertEqual(se_0, actual.kernel_one_d)
        self.assertEqual(name, actual.name)

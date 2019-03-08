from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from GPy.kern import RBF

from src.autoks.kernel import AKSKernel
from src.autoks.kernel_selection import KernelSelector
from src.evalg.selection import Selector


class TestKernelSelector(TestCase):

    def setUp(self):
        self.kernels = [AKSKernel(RBF(1)), AKSKernel(RBF(1))]
        self.scores = [21, 24]
        self.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        self.y_train = np.array([[5], [10]])

    def test__select(self):
        selector = Selector(1)
        selector.select = MagicMock()
        selector.select.return_value = np.array([self.kernels[1]])
        result = KernelSelector._select(self.kernels, self.scores, selector)
        self.assertEqual(result, [self.kernels[1]])

    def test_select_parents(self):
        parent_selector = Selector(1)
        parent_selector.select = MagicMock()
        parent_selector.select.return_value = np.array([self.kernels[1]])

        offspring_selector = MagicMock()
        kernel_pruner = MagicMock()
        ks = KernelSelector(parent_selector, offspring_selector, kernel_pruner)
        result = ks.select_parents(self.kernels, self.scores)
        self.assertEqual(result, [self.kernels[1]])

    def test_select_offspring(self):
        offspring_selector = Selector(1)
        offspring_selector.select = MagicMock()
        offspring_selector.select.return_value = np.array([self.kernels[1]])

        parent_selector = MagicMock()
        kernel_pruner = MagicMock()
        ks = KernelSelector(parent_selector, offspring_selector, kernel_pruner)
        result = ks.select_offspring(self.kernels, self.scores)
        self.assertEqual(result, [self.kernels[1]])

    def test_prune_candidates(self):
        kernel_pruner = Selector(1)
        kernel_pruner.select = MagicMock()
        kernel_pruner.select.return_value = np.array([self.kernels[1]])

        parent_selector = MagicMock()
        offspring_selector = MagicMock()
        ks = KernelSelector(parent_selector, offspring_selector, kernel_pruner)
        result = ks.prune_candidates(self.kernels, self.scores)
        self.assertEqual(result, [self.kernels[1]])

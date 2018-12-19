from unittest import TestCase

import numpy as np

from autoks.grammar import BaseGrammar


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

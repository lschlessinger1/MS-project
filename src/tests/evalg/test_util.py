from unittest import TestCase

from src.evalg import util


class TestUtil(TestCase):
    def test_swap(self):
        result = util.swap([1, 2, 3], 0, 2)
        self.assertEqual(result, [3, 2, 1])
        result = util.swap([1, 2, 3], 0, 0)
        self.assertEqual(result, [1, 2, 3])
        result = util.swap(['a', 'b', 3, 4, 5], 0, 4)
        self.assertEqual(result, [5, 'b', 3, 4, 'a'])

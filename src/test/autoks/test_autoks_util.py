from unittest import TestCase

from src.autoks.util import arg_sort, arg_unique, remove_duplicates


class TestUtil(TestCase):

    def test_arg_sort(self):
        items = [5, 2, 1, 3, 9]
        result = arg_sort(items)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, [2, 1, 3, 0, 4])

    def test_arg_unique(self):
        items = [5, 2, 2, 1, 1, 1, 3, 3, 9]
        result = arg_unique(items)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, [0, 1, 3, 6, 8])

    def test_remove_duplicates(self):
        data = [5, 2, 2, 1, 1, 1, 3, 3, 9]
        values = [str(d) for d in data]
        result = remove_duplicates(data, values)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ['5', '2', '1', '3', '9'])

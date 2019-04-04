from unittest import TestCase

from src.autoks.util import arg_sort, arg_unique, remove_duplicates, tokenize, flatten, remove_outer_parens, \
    join_operands, type_count


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

    def test_tokenize(self):
        result = tokenize([])
        self.assertListEqual(result, [])
        result = tokenize([1, 2, 3])
        self.assertListEqual(result, ['(', 1, 2, 3, ')'])
        result = tokenize([1, '+', [12, ['x'], 14], '+', 3])
        self.assertListEqual(result, ['(', 1, '+', ['(', 12, ['(', 'x', ')'], 14, ')'], '+', 3, ')'])

    def test_flatten(self):
        result = flatten([])
        self.assertListEqual(result, [])
        result = flatten(['(', 1, 2, 3, ')'])
        self.assertListEqual(result, ['(', 1, 2, 3, ')'])
        result = flatten(['(', 1, '+', ['(', 12, ['(', 'x', ')'], 14, ')'], '+', 3, ')'])
        self.assertListEqual(result, ['(', 1, '+', '(', 12, '(', 'x', ')', 14, ')', '+', 3, ')'])

    def test_remove_outer_parens(self):
        self.assertRaises(ValueError, remove_outer_parens, [])
        self.assertRaises(ValueError, remove_outer_parens, ['(', 1, 2, 3])
        self.assertRaises(ValueError, remove_outer_parens, [1, 2, 3, ')'])
        self.assertRaises(ValueError, remove_outer_parens, [1, '(', 3, ')'])
        result = remove_outer_parens(['(', 4, 5, 3, ')'])
        self.assertListEqual(result, [4, 5, 3])

    def test_join_operands(self):
        result = join_operands([1, 2, 3], '+')
        self.assertListEqual(result, [1, '+', 2, '+', 3])
        result = join_operands([1, [44, 77], 3], '+')
        self.assertListEqual(result, [1, '+', [44, 77], '+', 3])

    def test_type_count(self):
        result = type_count([1, 2, 3, '4'], str)
        self.assertEqual(1, result)
        result = type_count([1, 2, 3, '4'], int)
        self.assertEqual(3, result)

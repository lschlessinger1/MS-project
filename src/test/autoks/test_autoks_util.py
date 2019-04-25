from unittest import TestCase

from src.autoks.util import arg_sort, arg_unique, remove_duplicates, tokenize, flatten, remove_outer_parens, \
    join_operands, type_count, pretty_time_delta


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

    def test_remove_duplicates_multi_type(self):
        # simple example
        data = [1, 1, 1, 2, 3, 4, 'a', 'b', True]
        values = [10, 9, 8, '7', '6', False, 4, 3, 2]
        result = remove_duplicates(data, values)
        self.assertEqual(result, [10, '7', '6', False, 4, 3])

        with self.assertRaises(ValueError):
            remove_duplicates([1, 2, 3], ['1', 2])

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

    def test_pretty_time_delta(self):
        seconds = 172800  # 2 days
        actual = pretty_time_delta(seconds)
        expected = '2d0h0m0s'
        self.assertEqual(expected, actual)

        seconds = 172801  # 2 days, 1 second
        actual = pretty_time_delta(seconds)
        expected = '2d0h0m1s'
        self.assertEqual(expected, actual)

        seconds = 172861  # 2 days, 1 minute, 1 second
        actual = pretty_time_delta(seconds)
        expected = '2d0h1m1s'
        self.assertEqual(expected, actual)

        seconds = 176461  # 2 days, 1 hour, 1 minute, 1 second
        actual = pretty_time_delta(seconds)
        expected = '2d1h1m1s'
        self.assertEqual(expected, actual)

        seconds = 3661  # 1 hour, 1 minute, 1 second
        actual = pretty_time_delta(seconds)
        expected = '1h1m1s'
        self.assertEqual(expected, actual)

        seconds = 3601  # 1 hour, 1 second
        actual = pretty_time_delta(seconds)
        expected = '1h0m1s'
        self.assertEqual(expected, actual)

        seconds = 61
        actual = pretty_time_delta(seconds)
        expected = '1m1s'
        self.assertEqual(expected, actual)

        seconds = 33
        actual = pretty_time_delta(seconds)
        expected = '33s'
        self.assertEqual(expected, actual)

        seconds = 0.01
        actual = pretty_time_delta(seconds)
        expected = '10.0ms'
        self.assertEqual(expected, actual)

        seconds = 0.00001
        actual = pretty_time_delta(seconds)
        expected = '0.0ms'
        self.assertEqual(expected, actual)

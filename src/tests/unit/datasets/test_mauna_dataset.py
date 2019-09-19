import unittest

from src.datasets import MaunaDataset


class TestMaunaDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Mauna'
        actual = MaunaDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

import unittest

from src.datasets.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_name(self):
        expected = ''
        actual = Dataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

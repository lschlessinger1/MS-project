import unittest

from src.datasets import HousingDataset


class TestHousingDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Housing'
        actual = HousingDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

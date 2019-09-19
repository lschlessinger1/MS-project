import unittest

from src.datasets import SolarDataset


class TestSolarDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Solar'
        actual = SolarDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

import unittest

from src.datasets import AirlineDataset


class TestAirlineDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Airline'
        actual = AirlineDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

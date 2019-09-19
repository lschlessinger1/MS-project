import unittest

from src.datasets import MethaneDataset


class TestMethaneDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Methane'
        actual = MethaneDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

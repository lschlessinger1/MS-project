import unittest

from src.datasets import ConcreteDataset


class TestConcreteDataset(unittest.TestCase):

    def test_name(self):
        expected = 'Concrete'
        actual = ConcreteDataset().name
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

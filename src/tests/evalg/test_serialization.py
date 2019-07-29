import os
from unittest import TestCase

from src.evalg.serialization import Serializable


class TestSerialization(TestCase):

    def setUp(self) -> None:
        self.serializable = Serializable()

    def test_to_dict(self):
        actual = self.serializable.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertIn('__class__', actual)
        self.assertIn('__module__', actual)
        self.assertEqual(Serializable.__name__, actual['__class__'])
        self.assertEqual(Serializable.__module__, actual['__module__'])

    def test_from_dict(self):
        input_dict = self.serializable.to_dict()
        actual = Serializable.from_dict(input_dict)
        self.assertIsInstance(actual, Serializable)
        self.assertEqual(self.serializable.__dict__, actual.__dict__)

    def test__format_input_dict(self):
        input_dict = self.serializable.to_dict()
        actual = Serializable._format_input_dict(input_dict)
        self.assertIsInstance(actual, dict)
        self.assertDictEqual(input_dict, actual)

    def test_save(self):
        test_cases = (
            (False, 'Compress is false'),
            (True, 'Compress is true')
        )
        for compress, description in test_cases:
            with self.subTest(description=description):
                output_file_name = self.serializable.save(output_filename='tests', compress=compress)
                self.addCleanup(os.remove, output_file_name)

    def test_load(self):
        test_cases = (
            (False, 'Compress is false'),
            (True, 'Compress is true')
        )
        for compress, description in test_cases:
            with self.subTest(description=description):
                output_file_name = self.serializable.save(output_filename='tests', compress=compress)
                self.addCleanup(os.remove, output_file_name)

                new_serializable = Serializable.load(output_file_name)

                self.assertIsInstance(new_serializable, Serializable)
                self.assertDictEqual(self.serializable.to_dict(), new_serializable.to_dict())

import os
from unittest import TestCase

from src.evalg.serialization import Serializable


class TestSerialization(TestCase):

    def setUp(self) -> None:
        self.serializable = Serializable()

    def test_save(self):
        test_cases = (
            (False, 'Compress is false'),
            (True, 'Compress is true')
        )
        for compress, description in test_cases:
            with self.subTest(description=description):
                output_file_name = self.serializable.save(output_filename='test', compress=compress)
                self.addCleanup(os.remove, output_file_name)

    def test_load(self):
        test_cases = (
            (False, 'Compress is false'),
            (True, 'Compress is true')
        )
        for compress, description in test_cases:
            with self.subTest(description=description):
                output_file_name = self.serializable.save(output_filename='test', compress=compress)
                self.addCleanup(os.remove, output_file_name)

                new_serializable = Serializable.load(output_file_name)

                self.assertIsInstance(new_serializable, Serializable)
                self.assertDictEqual(self.serializable.to_dict(), new_serializable.to_dict())

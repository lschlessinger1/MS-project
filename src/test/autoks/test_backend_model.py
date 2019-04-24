from unittest import TestCase
from unittest.mock import MagicMock

from src.autoks.backend.model import cov_parsimony_pressure


class TestBackendModel(TestCase):

    def test_cov_parsimony_pressure(self):
        model = MagicMock()
        model._size_transformed.return_value = 2
        model.log_likelihood.return_value = 8
        model_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        model_sizes = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = cov_parsimony_pressure(model, model_scores, model_sizes)
        self.assertIsInstance(result, float)
        self.assertEqual(10.0, result)

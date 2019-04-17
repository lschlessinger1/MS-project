from unittest import TestCase

import numpy as np

from src.autoks.distance.util import probability_samples


class TestDistanceUtil(TestCase):

    def test_probability_samples(self):
        m = 15
        n = 10
        result = probability_samples(max_num_hyperparameters=m, num_samples=n)
        assert isinstance(result, np.ndarray)
        assert result.shape == (n, m)

from unittest import TestCase

from src.autoks.backend.prior import PRIOR_DICT
from src.autoks.core.prior import PriorDist
from src.evalg.serialization import Serializable


class TestPrior(TestCase):

    def setUp(self) -> None:
        self.input_dict = {'mu': 0, 'sigma': 1}

    def test_from_str(self):
        key = 'GAUSSIAN'
        prior = PriorDist.from_prior_str(key, self.input_dict)
        self.assertIsInstance(prior, PriorDist)
        self.assertEqual(PRIOR_DICT[key], prior.raw_prior.__class__)

    def test_to_dict(self):
        prior = PriorDist.from_prior_str('GAUSSIAN', self.input_dict)
        actual = prior.to_dict()
        self.assertIsInstance(actual, dict)
        self.assertIn('__class__', actual)
        self.assertIn('__module__', actual)
        self.assertIn("raw_prior_cls", actual)
        self.assertIn("raw_prior_module", actual)
        self.assertIn("raw_prior_args", actual)
        self.assertEqual(actual["raw_prior_cls"], prior.raw_prior.__class__.__name__)
        self.assertEqual(actual["raw_prior_module"], prior.raw_prior.__module__)
        self.assertEqual(actual["raw_prior_args"], self.input_dict)

    def test_from_dict(self):
        test_cases = (PriorDist, Serializable)
        for cls in test_cases:
            with self.subTest(name=cls.__name__):
                key = 'GAUSSIAN'
                prior = PriorDist.from_prior_str(key, self.input_dict)
                actual = cls.from_dict(prior.to_dict())
                self.assertIsInstance(actual, PriorDist)
                self.assertEqual(self.input_dict, actual._raw_prior_args)
                self.assertEqual(PRIOR_DICT[key], actual._raw_prior_cls)

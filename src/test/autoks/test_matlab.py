from unittest import TestCase

import numpy as np

has_matlab_lib = False
try:
    import matlab.engine
    from matlab.engine.matlabengine import MatlabEngine
    from src.autoks.distance.matlab import prob_samples_matlab, start_matlab, quit_matlab

    has_matlab_lib = True
    print('Starting MATLAB engine...')
    eng = start_matlab()
except ImportError:
    from warnings import warn

    warn('Could not import MATLAB engine. Skipping MATLAB unit tests.')


class TestMatlab(TestCase):

    def setUp(self) -> None:
        if not has_matlab_lib:
            self.skipTest('MATLAB engine could not be imported.')

    def test_prob_samples_matlab(self):
        m = 15
        n = 10
        result = prob_samples_matlab(eng, max_num_hyperparameters=m, num_samples=n)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (n, m))

    @classmethod
    def tearDownClass(cls) -> None:
        if has_matlab_lib:
            quit_matlab(eng)

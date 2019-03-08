import numpy as np

from src.autoks.kernel import AKSKernel


class AcquisitionFunction:
    """Abstract base class for all acquisition functions."""

    def score(self, kernel: AKSKernel, x_train: np.ndarray, y_train: np.ndarray, hyperpriors=None) -> float:
        """Acquisition function score.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        raise NotImplementedError('Must be implemented in a child class')


# Acquisition Functions

class UniformScorer(AcquisitionFunction):
    CONST_SCORE = 1

    def score(self, kernel: AKSKernel, x_train: np.ndarray, y_train: np.ndarray, hyperpriors=None) -> float:
        """Same score for all kernels.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return self.CONST_SCORE


class ExpectedImprovement(AcquisitionFunction):

    def score(self, kernel: AKSKernel, x_train: np.ndarray, y_train: np.ndarray, hyperpriors=None) -> float:
        """Expected improvement (EI) acquisition function

        This acquisition function takes a model (kernel and hyperpriors) and computes expected improvement using
        the posterior Hellinger squared exponential covariance between models conditioned on the training data

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        pass

from typing import Optional, List

import numpy as np
from GPy.core.parameterization.priors import Prior

from src.autoks.kernel import AKSKernel


class AcquisitionFunction:
    """Abstract base class for all acquisition functions."""

    def score(self,
              kernel: AKSKernel,
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
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
    CONST_SCORE: float = 1

    def score(self,
              kernel: AKSKernel,
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
        """Same score for all kernels.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return self.CONST_SCORE


class ExpectedImprovement(AcquisitionFunction):

    def score(self,
              kernel: AKSKernel,
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
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


class RandomScorer(AcquisitionFunction):

    def score(self,
              kernel: AKSKernel,
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
        """Random acquisition function

        This acquisition function returns a random score in the half-open interval [0.0, 1.0).

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return np.random.random()


class ParamProportionalScorer(AcquisitionFunction):

    def score(self, kernel: AKSKernel,
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
        """Score proportional to number of kernel hyperparameters.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return -kernel.kernel.num_params  # return the negative because we want to minimize this

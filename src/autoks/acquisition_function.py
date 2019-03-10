from typing import Optional, List

import numpy as np
from GPy.core.parameterization.priors import Prior

from src.autoks.kernel import AKSKernel, n_base_kernels


class AcquisitionFunction:
    """Abstract base class for all acquisition functions."""

    @staticmethod
    def score(kernel: AKSKernel,
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

    @staticmethod
    def score(kernel: AKSKernel,
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
        return UniformScorer.CONST_SCORE


class ExpectedImprovement(AcquisitionFunction):

    @staticmethod
    def score(kernel: AKSKernel,
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

    @staticmethod
    def score(kernel: AKSKernel,
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

    @staticmethod
    def score(kernel: AKSKernel,
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


class OperandProportionalScorer(AcquisitionFunction):

    @staticmethod
    def score(kernel: AKSKernel, x_train: np.ndarray, y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
        """Score proportional to the number of 1D kernels (operands).

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        return -n_base_kernels(kernel.kernel)  # return the negative because we want to minimize this


class KernComplexityProportionalScorer(AcquisitionFunction):
    @staticmethod
    def score(kernel: AKSKernel, x_train: np.ndarray, y_train: np.ndarray,
              hyperpriors: Optional[List[Prior]] = None) -> float:
        """Score proportional to the complexity of a kernel

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :return:
        """
        param_score = ParamProportionalScorer.score(kernel, x_train, y_train, hyperpriors)
        operand_score = OperandProportionalScorer.score(kernel, x_train, y_train, hyperpriors)
        return param_score + operand_score

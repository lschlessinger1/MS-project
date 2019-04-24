from typing import Optional, List

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from src.autoks.backend.kernel import n_base_kernels
from src.autoks.core.gp_model import GPModel
from src.autoks.hyperprior import Hyperpriors


class AcquisitionFunction:
    """Abstract base class for all acquisition functions."""

    @staticmethod
    def score(
            ind: int,
            all_kernels: List[GPModel],
            x_train: np.ndarray,
            y_train: np.ndarray,
            hyperpriors: Optional[Hyperpriors] = None,
            surrogate_model: Optional = None,
            **kwargs) -> float:
        """Acquisition function score.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        raise NotImplementedError('Must be implemented in a child class')


# Acquisition Functions

class UniformScorer(AcquisitionFunction):
    CONST_SCORE: float = 1

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> float:
        """Same score for all kernels.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        return UniformScorer.CONST_SCORE


class ExpectedImprovement(AcquisitionFunction):

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model=None,
              **kwargs) -> float:
        """Expected improvement (EI) acquisition function

        This acquisition function takes a model (kernel and hyperpriors) and computes expected improvement using
        the posterior Hellinger squared exponential covariance between models conditioned on the training data

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        x_test = np.array([ind])[:, None]
        mu, cov = surrogate_model.predict(x_test)
        y_max = surrogate_model.predict(surrogate_model.X)[0].max()

        # Ensure cov > 0.
        cov[cov < 0] = 0
        sigma = np.sqrt(cov)

        improvement = mu - y_max
        u = improvement / sigma
        ei = improvement * norm.cdf(u) + sigma * norm.pdf(u)
        ei[sigma == 0.0] = 0.0
        return ei[0, 0]


class ExpectedImprovementPerSec(AcquisitionFunction):

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              durations=None,
              n_hyperparams=None,
              **kwargs) -> float:
        kernel = all_kernels[ind]
        scorer = ExpectedImprovement()
        ei = scorer.score(ind, all_kernels, x_train, y_train, hyperpriors, surrogate_model)
        reg = LinearRegression()
        x = np.array(n_hyperparams)[:, None]
        y = np.log(durations)
        reg.fit(x, y)
        t = reg.predict(np.array([[kernel.kernel.size]]))
        eps = np.spacing(1)
        t[t <= 0] = eps
        return ei / t[0]


class RandomScorer(AcquisitionFunction):

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> float:
        """Random acquisition function

        This acquisition function returns a random score in the half-open interval [0.0, 1.0).

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        return np.random.random()


class ParamProportionalScorer(AcquisitionFunction):

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> float:
        """Score proportional to number of kernel hyperparameters.

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        kernel = all_kernels[ind]
        return -kernel.kernel.size  # return the negative because we want to minimize this


class OperandProportionalScorer(AcquisitionFunction):

    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> float:
        """Score proportional to the number of 1D kernels (operands).

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        kernel = all_kernels[ind]
        return -n_base_kernels(kernel.kernel)  # return the negative because we want to minimize this


class KernComplexityProportionalScorer(AcquisitionFunction):
    @staticmethod
    def score(ind: int,
              all_kernels: List[GPModel],
              x_train: np.ndarray,
              y_train: np.ndarray,
              hyperpriors: Optional[Hyperpriors] = None,
              surrogate_model: Optional = None,
              **kwargs) -> float:
        """Score proportional to the complexity of a kernel

        :param kernel:
        :param x_train:
        :param y_train:
        :param hyperpriors:
        :param surrogate_model:
        :return:
        """
        param_score = ParamProportionalScorer.score(ind, all_kernels, x_train, y_train, hyperpriors)
        operand_score = OperandProportionalScorer.score(ind, all_kernels, x_train, y_train, hyperpriors)
        return param_score + operand_score

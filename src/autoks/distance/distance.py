from typing import Tuple

import numpy as np
from GPy.core.parameterization.priors import Prior
from GPy.kern import Kern
from numpy.linalg import LinAlgError

from src.autoks.distance.util import probability_samples, prior_sample
# Adapted from Malkomes, 2016
# Bayesian optimization for automated model selection (BOMS)
# https://github.com/gustavomalkomes/automated_model_selection
from src.autoks.kernel import get_priors


class DistanceBuilder:
    """DistanceBuilder Build distance matrix between models."""

    def __init__(self,
                 noise_prior: Prior,
                 num_samples: int,
                 max_num_hyperparameters: int,
                 max_num_kernels: int,
                 active_models,
                 initial_model_indices,
                 data_X: np.ndarray):
        self.num_samples = num_samples
        self.max_num_hyperparameters = max_num_hyperparameters
        self.max_num_kernels = max_num_kernels

        self.probability_samples = probability_samples(max_num_hyperparameters=self.max_num_hyperparameters,
                                                       num_samples=self.num_samples)

        noise_prior = np.array([noise_prior])
        noise_samples = prior_sample(noise_prior, self.probability_samples)
        self.hyperparameter_data_noise_samples = np.exp(noise_samples)
        self._average_distance = np.full((self.max_num_kernels, self.max_num_kernels), np.nan)
        np.fill_diagonal(self._average_distance, 0)
        self.precompute_information(active_models, initial_model_indices, data_X)

    def precompute_information(self,
                               active_models,
                               new_candidates_indices,
                               data_X: np.ndarray) -> None:
        """Precompute distance information for each new candidate.

        :param active_models:
        :param new_candidates_indices:
        :param data_X:
        :return:
        """
        # TODO: test correctness
        for i in new_candidates_indices:
            covariance = active_models.models[i].covariance
            precomputed_info = self.create_precomputed_info(covariance, data_X)
            active_models.models[i].set_precomputed_info(precomputed_info)

    def update(self,
               active_models,
               new_candidates_indices,
               all_candidates_indices,
               selected_indices,
               data_X: np.ndarray) -> None:
        """Update average distance between models.

        :param active_models:
        :param new_candidates_indices:
        :param all_candidates_indices:
        :param selected_indices:
        :param data_X:
        :return:
        """
        # TODO: test correctness
        # First step is to precompute information for the new candidate models
        self.precompute_information(active_models, new_candidates_indices, data_X)

        # Second step is to compute the distance between the trained models vs candidate models.
        new_evaluated_models = selected_indices[-1]
        all_old_candidates_indices = np.setdiff1d(all_candidates_indices, new_candidates_indices)

        # i) new evaluated models vs all old candidates.
        self.compute_distance(active_models, new_evaluated_models, all_old_candidates_indices)

        # ii) new candidate models vs all trained models
        self.compute_distance(active_models, selected_indices, new_candidates_indices)

    def get_kernel(self, index: int) -> np.ndarray:
        """

        :param index:
        :return:
        """
        # TODO: test correctness
        return self._average_distance[:index, :index]

    def compute_distance(self,
                         active_models,
                         indices_i,
                         indices_j) -> None:
        raise NotImplementedError

    def create_precomputed_info(self,
                                covariance,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class HellingerDistanceBuilder(DistanceBuilder):
    """HellingerDistanceBuilder builds distances based on the Hellinger distance between
    the model's Gram matrices.
    """

    def __init__(self, noise_prior, num_samples, max_num_hyperparameters, max_num_kernels,
                 active_models, initial_model_indices, data_X):
        super().__init__(noise_prior, num_samples, max_num_hyperparameters, max_num_kernels,
                         active_models, initial_model_indices, data_X)

    @staticmethod
    def hellinger_distance(data_i, data_j):
        pass

    def compute_distance(self, active_models, indices_i, indices_j) -> None:
        pass

    def create_precomputed_info(self,
                                covariance: Kern,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = data_X.shape[0]
        tolerance = 1e-6

        log_det = np.full(self.num_samples, np.nan)
        mini_gram_matrices = np.full((n, n, self.num_samples), np.nan)

        cov_priors = get_priors(covariance)
        hyperparameters = prior_sample(cov_priors, self.probability_samples)
        # For numerical stability
        hyperparameters[hyperparameters == 0] = 1e-9

        for i in range(hyperparameters.shape[0]):
            hyp = hyperparameters[i, :]
            lmbda = self.hyperparameter_data_noise_samples[i]

            covariance[:] = hyp
            k = covariance.K(data_X, data_X)
            k = k + lmbda * np.eye(k.shape[0])

            mini_gram_matrices[:, :, i] = k
            chol_k = chol_safe(k, tolerance)
            log_det[i] = 2 * np.sum(np.log(np.diag(chol_k)), axis=0)

        return log_det, mini_gram_matrices


def fix_numerical_problem(k: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """

    :param k:
    :param tolerance:
    :return:
    """
    d, v = np.linalg.eig(k)
    new_diagonal = d
    new_diagonal[new_diagonal < tolerance] = tolerance
    new_diagonal = np.diag(new_diagonal)
    k = v @ new_diagonal @ v.T
    k = (k + k.T) / 2
    chol_k = np.linalg.cholesky(k).T
    return chol_k


def chol_safe(k: np.ndarray,
              tolerance: float) -> np.ndarray:
    """Safe Cholesky decomposition.

    k: covariance matrix (n x n)
    """
    try:
        chol_k = np.linalg.cholesky(k).T
    except LinAlgError:
        # Decomposition failed, k may not be positive-definite.
        # Try to recover by making the covariance matrix positive-definite.
        chol_k = fix_numerical_problem(k, tolerance)
    return chol_k

from typing import Tuple, List

import numpy as np
from GPy.core.parameterization.priors import Prior, Gaussian
from GPy.kern import Kern
from numpy.linalg import LinAlgError
from statsmodels.stats.correlation_tools import cov_nearest

from src.autoks.backend.kernel import get_priors
from src.autoks.core.active_set import ActiveSet
from src.autoks.distance.util import probability_samples, prior_sample

# Adapted from Malkomes, 2016
# Bayesian optimization for automated model selection (BOMS)
# https://github.com/gustavomalkomes/automated_model_selection


# For now this represents the active set class
ActiveModels = ActiveSet


class DistanceBuilder:
    """DistanceBuilder Build distance matrix between models."""
    hyperparameter_data_noise_samples: np.ndarray
    _average_distance: np.ndarray

    def __init__(self,
                 noise_prior: Prior,
                 num_samples: int,
                 max_num_hyperparameters: int,
                 max_num_kernels: int,
                 active_models: ActiveModels,
                 initial_model_indices: List[int],
                 data_X: np.ndarray):
        self.num_samples = num_samples
        self.max_num_hyperparameters = max_num_hyperparameters
        self.max_num_kernels = max_num_kernels

        self.probability_samples = probability_samples(max_num_hyperparameters=self.max_num_hyperparameters,
                                                       num_samples=self.num_samples)

        # FIXME: This forces the noise prior to be gaussian because we then exponentiate it, making it a Log-Gaussian
        assert noise_prior.__class__ == Gaussian
        noise_prior = np.array([noise_prior])
        noise_samples = prior_sample(noise_prior, self.probability_samples)
        self.hyperparameter_data_noise_samples = np.exp(noise_samples)

        self._average_distance = np.full((self.max_num_kernels, self.max_num_kernels), np.nan)
        np.fill_diagonal(self._average_distance, 0)
        self.precompute_information(active_models, initial_model_indices, data_X)

    def precompute_information(self,
                               active_models: ActiveModels,
                               new_candidates_indices: List[int],
                               data_X: np.ndarray) -> None:
        """Precompute distance information for each new candidate.

        :param active_models:
        :param new_candidates_indices:
        :param data_X:
        :return:
        """
        for i in new_candidates_indices:
            # covariance = active_models.models[i].covariance
            covariance = active_models.models[i].kernel
            precomputed_info = self.create_precomputed_info(covariance, data_X)
            # active_models.models[i].info = precomputed_info
            active_models.models[i].info = precomputed_info

    # TODO: remove and refactor original update method
    def update_multiple(self,
                        active_models: ActiveModels,
                        new_candidates_ind: List[int],
                        all_candidates_ind: List[int],
                        old_selected_ind: List[int],
                        new_selected_ind: List[int],
                        data_X: np.ndarray) -> None:
        # First step is to precompute information for the new candidate models
        self.precompute_information(active_models, new_candidates_ind, data_X)

        # Second step is to compute the distance between the trained models vs candidate models.
        new_evaluated_models = new_selected_ind
        all_old_candidates_indices = np.setdiff1d(all_candidates_ind, new_candidates_ind)

        # compute distance between evaluated kernels
        if len(new_evaluated_models) > 1:
            self.compute_distance(active_models, new_evaluated_models, new_evaluated_models)

        # i) new evaluated models vs all old candidates.
        self.compute_distance(active_models, new_evaluated_models, list(all_old_candidates_indices.tolist()))

        # ii) new candidate models vs all trained models
        all_selected = old_selected_ind + new_selected_ind
        self.compute_distance(active_models, all_selected, new_candidates_ind)

    def update(self,
               active_models: ActiveModels,
               new_candidates_indices: List[int],
               all_candidates_indices: List[int],
               selected_indices: List[int],
               data_X: np.ndarray) -> None:
        """Update average distance between models.

        :param active_models:
        :param new_candidates_indices:
        :param all_candidates_indices:
        :param selected_indices:
        :param data_X:
        :return:
        """
        # First step is to precompute information for the new candidate models
        self.precompute_information(active_models, new_candidates_indices, data_X)

        # Second step is to compute the distance between the trained models vs candidate models.
        new_evaluated_models = selected_indices[-1]
        all_old_candidates_indices = np.setdiff1d(all_candidates_indices, new_candidates_indices)

        # i) new evaluated models vs all old candidates.
        self.compute_distance(active_models, [new_evaluated_models], list(all_old_candidates_indices.tolist()))

        # ii) new candidate models vs all trained models
        self.compute_distance(active_models, selected_indices, new_candidates_indices)

    def get_kernel(self, index: int) -> np.ndarray:
        """

        :param index:
        :return:
        """
        return self._average_distance[:index, :index]

    def compute_distance(self,
                         active_models: ActiveModels,
                         indices_i: List[int],
                         indices_j: List[int]) -> None:
        raise NotImplementedError

    def create_precomputed_info(self,
                                covariance: Kern,
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
    def hellinger_distance(log_det_i: np.ndarray,
                           mini_gram_matrices_i: np.ndarray,
                           log_det_j: np.ndarray,
                           mini_gram_matrices_j: np.ndarray,
                           tol: float = 0.02) -> float:
        are_different = np.abs(log_det_i - log_det_j) > tol
        indices = np.arange(are_different.size)
        logdet_p_and_q = log_det_i
        for i in indices[are_different]:
            p_K = mini_gram_matrices_i[:, :, i]
            q_K = mini_gram_matrices_j[:, :, i]
            p_and_q_kernels = 0.5 * (p_K + q_K)
            chol_p_and_q = chol_safe(p_and_q_kernels, tol)
            logdet_p_and_q[i] = 2 * np.sum(np.log(np.diag(chol_p_and_q)), axis=0)

        # Compute log distance.
        log_det_sum = log_det_i + log_det_j
        log_hellinger = 0.25 * log_det_sum - 0.5 * logdet_p_and_q

        # Exponentiate.
        hellinger = 1 - np.exp(log_hellinger)
        distance = np.mean(hellinger, axis=0)
        return float(distance)

    def compute_distance(self,
                         active_models: ActiveModels,
                         indices_i: List[int],
                         indices_j: List[int]) -> None:
        # TODO: test correctness
        for i in indices_i:
            for j in indices_j:
                dist = self.hellinger_distance(*active_models.models[i].info, *active_models.models[j].info)
                self._average_distance[i, j] = dist
                self._average_distance[j, i] = dist

    def create_precomputed_info(self,
                                covariance: Kern,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = data_X.shape[0]
        tolerance = 1e-6

        log_det = np.full(self.num_samples, np.nan)
        mini_gram_matrices = np.full((n, n, self.num_samples), np.nan)

        cov_priors = get_priors(covariance)
        hyperparameters = prior_sample(cov_priors, self.probability_samples)

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
    k = cov_nearest(k, threshold=tolerance)
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

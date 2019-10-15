from typing import Tuple, List

import numpy as np
from GPy.core.parameterization.priors import Prior, Gaussian
from numpy.linalg import LinAlgError
from statsmodels.stats.correlation_tools import cov_nearest

from src.autoks.backend.kernel import get_priors
from src.autoks.core.active_set import ActiveSet
from src.autoks.core.covariance import Covariance
from src.autoks.distance import util

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

        self.probability_samples = util.probability_samples(max_num_hyperparameters=self.max_num_hyperparameters,
                                                            num_samples=self.num_samples)

        # FIXME: This forces the noise prior to be gaussian because we then exponentiate it, making it a Log-Gaussian
        assert noise_prior.__class__ == Gaussian
        noise_prior = np.array([noise_prior])
        noise_samples = util.prior_sample(noise_prior, self.probability_samples)
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
            covariance = active_models.models[i].covariance
            precomputed_info = self.create_precomputed_info(covariance, data_X)
            active_models.models[i].info = precomputed_info

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

    @staticmethod
    def metric(data_i, data_j, **kwargs) -> float:
        raise NotImplementedError

    def compute_distance(self,
                         active_models: ActiveModels,
                         indices_i: List[int],
                         indices_j: List[int]) -> None:
        for i in indices_i:
            for j in indices_j:
                dist = self.metric(active_models.models[i].info, active_models.models[j].info)
                self._average_distance[i, j] = dist
                self._average_distance[j, i] = dist

    def create_precomputed_info(self,
                                covariance: Covariance,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class HellingerDistanceBuilder(DistanceBuilder):
    """HellingerDistanceBuilder builds distances based on the Hellinger distance between
    the model's Gram matrices.
    """

    @staticmethod
    def metric(data_i, data_j, **kwargs) -> float:
        return HellingerDistanceBuilder.hellinger_distance(*data_i, *data_j, **kwargs)

    @staticmethod
    def hellinger_distance(log_det_i: np.ndarray,
                           mini_gram_matrices_i: np.ndarray,
                           log_det_j: np.ndarray,
                           mini_gram_matrices_j: np.ndarray,
                           tol: float = 0.02) -> float:
        """Hellinger distance between two multivariate Gaussian distributions with zero means zero.

        https://en.wikipedia.org/wiki/Hellinger_distance
        """
        are_different = np.abs(log_det_i - log_det_j) > tol
        indices = np.arange(are_different.size)
        logdet_p_and_q = log_det_i.copy()
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

        # for numerical stability, clip distance to [0, 1] before taking sqrt
        distance = np.clip(distance, 0, 1)
        distance = np.sqrt(distance)

        return float(distance)

    def create_precomputed_info(self,
                                covariance: Covariance,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = data_X.shape[0]
        tolerance = 1e-6

        log_det = np.full(self.num_samples, np.nan)
        mini_gram_matrices = np.full((n, n, self.num_samples), np.nan)

        cov_priors = get_priors(covariance.raw_kernel)
        hyperparameters = util.prior_sample(cov_priors, self.probability_samples)

        for i in range(hyperparameters.shape[0]):
            hyp = hyperparameters[i, :]
            lmbda = self.hyperparameter_data_noise_samples[i]

            covariance.raw_kernel[:] = hyp
            k = covariance.raw_kernel.K(data_X, data_X)
            k = k + lmbda * np.eye(k.shape[0])

            mini_gram_matrices[:, :, i] = k
            chol_k = chol_safe(k, tolerance)
            log_det[i] = 2 * np.sum(np.log(np.diag(chol_k)), axis=0)

        return log_det, mini_gram_matrices


class FrobeniusDistanceBuilder(DistanceBuilder):

    def __init__(self, noise_prior: Prior, num_samples: int, max_num_hyperparameters: int, max_num_kernels: int,
                 active_models: ActiveModels, initial_model_indices: List[int], data_X: np.ndarray):
        super().__init__(noise_prior, num_samples, max_num_hyperparameters, max_num_kernels, active_models,
                         initial_model_indices, data_X)

    @staticmethod
    def metric(data_i, data_j, **kwargs) -> float:
        return FrobeniusDistanceBuilder.frobenius_distance(data_i, data_j)

    @staticmethod
    def frobenius_distance(a: np.ndarray,
                           b: np.ndarray) -> float:
        """Average Frobenius distance between a vs b."""
        distance = np.mean(np.sqrt(np.sum((a - b) ** 2, axis=0)))
        return float(distance)

    def create_precomputed_info(self,
                                covariance: Covariance,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = data_X.shape[0]

        vectors = np.full((n ** 2, self.num_samples), np.nan, dtype=np.float32)

        cov_priors = get_priors(covariance.raw_kernel)
        hyperparameters = util.prior_sample(cov_priors, self.probability_samples)
        for i in range(hyperparameters.shape[0]):
            hyp = hyperparameters[i, :]
            noise_var = self.hyperparameter_data_noise_samples[i]
            covariance.raw_kernel[:] = hyp
            prior_covariance = covariance.raw_kernel.K(data_X, data_X)
            prior_covariance += noise_var * np.eye(prior_covariance.shape[0])
            vectors[:, i] = prior_covariance.reshape(n * n).copy()

        return vectors


class CorrelationDistanceBuilder(DistanceBuilder):

    @staticmethod
    def metric(data_i, data_j, **kwargs) -> float:
        return CorrelationDistanceBuilder.correlation_distance(data_i, data_j)

    @staticmethod
    def correlation_distance(a: np.ndarray,
                             b: np.ndarray) -> float:
        """Average correlation distance between a vs b."""
        a_mean = np.mean(a, axis=0)
        b_mean = np.mean(b, axis=0)
        a_centered = a - a_mean
        b_centered = b - b_mean
        # Batch dot product: sum of dot products for all vectors in a and
        dot_prod = np.einsum('ij,ji->i', a_centered.T, b_centered)
        a_norm = np.linalg.norm(a_centered, axis=0)
        b_norm = np.linalg.norm(b_centered, axis=0)
        correlation = dot_prod / (a_norm * b_norm)

        # For numerical stability, clip distance to [0, 1] before taking sqrt.
        correlation = np.clip(correlation, 0, 1)

        # Ordinally equivalent to the angular distance (arccos(correlation)).
        # See Metric distances derived from cosine similarity and Pearson and
        # Spearman correlations, Dongen & Enright (2012).
        correlation_dist = np.sqrt(0.5 * (1 - correlation))

        distance = np.mean(correlation_dist, axis=0)

        return float(distance)

    def create_precomputed_info(self,
                                covariance: Covariance,
                                data_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = data_X.shape[0]

        vectors = np.full((n ** 2, self.num_samples), np.nan, dtype=np.float32)

        cov_priors = get_priors(covariance.raw_kernel)
        hyperparameters = util.prior_sample(cov_priors, self.probability_samples)
        for i in range(hyperparameters.shape[0]):
            hyp = hyperparameters[i, :]
            noise_var = self.hyperparameter_data_noise_samples[i]
            covariance.raw_kernel[:] = hyp
            prior_covariance = covariance.raw_kernel.K(data_X, data_X)
            prior_covariance += noise_var * np.eye(prior_covariance.shape[0])
            vectors[:, i] = prior_covariance.reshape(n * n).copy()

        return vectors


def fix_numerical_problem(k: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """

    :param k:
    :param tolerance:
    :return:
    """
    k = cov_nearest(k, threshold=tolerance)
    cholesky_k = np.linalg.cholesky(k).T
    return cholesky_k


def chol_safe(k: np.ndarray,
              tolerance: float) -> np.ndarray:
    """Safe Cholesky decomposition.

    k: covariance matrix (n x n)
    """
    try:
        cholesky_k = np.linalg.cholesky(k).T
    except LinAlgError:
        # Decomposition failed, k may not be positive-definite.
        # Try to recover by making the covariance matrix positive-definite.
        cholesky_k = fix_numerical_problem(k, tolerance)
    return cholesky_k

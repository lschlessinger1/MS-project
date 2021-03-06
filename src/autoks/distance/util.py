import numpy as np
from GPy.core.parameterization.priors import LogGaussian, Gaussian
from scipy.stats import norm

from src.autoks.distance.sampling import sample


def probability_samples(max_num_hyperparameters: int = 40,
                        num_samples: int = 20,
                        sampling_method: str = 'generalized_halton') -> np.ndarray:
    """Sample from a given sampling method.

    :param max_num_hyperparameters:
    :param num_samples:
    :param sampling_method:
    :return: a num_samples x max_num_hyperparameters array
    """
    return sample(sampling_method, num_samples, max_num_hyperparameters)


def prior_sample(priors: np.ndarray,
                 prob_samples: np.ndarray) -> np.ndarray:
    # for each prior, call the PPF (inverse of CDF)
    # allowed prior types: Gaussian, LogGaussian
    gaussian_ind = [i for (i, prior) in enumerate(priors) if type(prior) == Gaussian]
    log_gaussian_ind = [i for (i, prior) in enumerate(priors) if type(prior) == LogGaussian]

    if len(gaussian_ind) + len(log_gaussian_ind) < priors.size:
        raise ValueError('Unsupported prior found in priors')

    hyps = np.full((prob_samples.shape[0], priors.size), np.nan)

    if gaussian_ind:
        gaussian_samples = prior_sample_gaussian(priors[gaussian_ind], prob_samples)
        hyps[:, gaussian_ind] = gaussian_samples

    if log_gaussian_ind:
        log_gaussian_samples = prior_sample_log_gaussian(priors[log_gaussian_ind], prob_samples)
        hyps[:, log_gaussian_ind] = log_gaussian_samples

    return hyps


def prior_sample_gaussian(priors: np.ndarray,
                          prob_samples: np.ndarray) -> np.ndarray:
    """
    Return num_samples x priors.shape[0] array
    """
    num_samples = prob_samples.shape[0]
    num_hyp = priors.size
    prior_mean = np.zeros(num_hyp)
    prior_std = np.zeros(num_hyp)
    for i in range(num_hyp):
        prior = priors[i]
        if not isinstance(prior, Gaussian):
            raise ValueError('All priors must be Gaussian')
        prior_mean[i] = prior.mu
        prior_std[i] = prior.sigma

    hyps = norm.ppf(prob_samples[:, :num_hyp],
                    np.tile(prior_mean, (num_samples, 1)),
                    np.tile(prior_std, (num_samples, 1)))
    return hyps


def prior_sample_log_gaussian(priors: np.ndarray,
                              prob_samples: np.ndarray) -> np.ndarray:
    """
    Return num_samples x priors.shape[0] array
    """
    return np.exp(prior_sample_gaussian(priors, prob_samples))

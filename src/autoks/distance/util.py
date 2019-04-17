import numpy as np
from scipy.stats import norm

from src.autoks.distance.sobol import sobol_sample


def probability_samples(max_num_hyperparameters: int = 40,
                        num_samples: int = 20) -> np.ndarray:
    """Sample from low discrepancy Sobol sequence.

    :param max_num_hyperparameters:
    :param num_samples:
    :return: a num_samples x max_num_hyperparameters array
    """
    # FIXME:
    #  - should support up to 50 spatial dimensions (instead of 40)
    #  - leap is causing sample to be very slow (therefore, set it to 0 instead of 100)
    #  - scramble should be the Matouse-Affine-Owen scramble method
    return sobol_sample(num_samples, max_num_hyperparameters, skip=1000, leap=0, scramble=True)


def prior_sample(priors: np.ndarray,
                 prob_samples: np.ndarray) -> np.ndarray:
    """
    Return num_samples x priors.shape[0] array
    """
    num_samples = prob_samples.shape[0]
    num_hyp = priors.size
    prior_mean = np.zeros((1, priors.size))
    prior_std = np.zeros((1, priors.size))
    for i in range(num_hyp):
        prior = priors[i]
        prior_mean[0, i] = prior.mu
        prior_std[0, i] = prior.sigma

    hyps = norm.ppf(prob_samples[:, :num_hyp],
                    np.tile(prior_mean, (num_samples, 1)),
                    np.tile(prior_std, (num_samples, 1)))
    return hyps

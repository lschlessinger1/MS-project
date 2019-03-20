from typing import Dict

from GPy.core.parameterization.priors import LogGaussian


def boms_hyperpriors() -> Dict[str, Dict[str, LogGaussian]]:
    """

    Malkomes et al., 2016

    :return: The hyperparameter priors used in BOMS.
    """
    prior_l = LogGaussian(mu=0.1, sigma=0.7 ** 2)  # length scales (period of a periodic covariance, etc.)
    prior_sigma = LogGaussian(mu=0.4, sigma=0.7 ** 2)  # signal variance
    prior_sigma_n = LogGaussian(mu=0.1, sigma=1 ** 2)  # observation noise
    prior_alpha = LogGaussian(mu=0.05, sigma=0.7 ** 2)  # α parameter of the rational quadratic covariance
    prior_l_p = LogGaussian(mu=2, sigma=0.7 ** 2)  # the "length scale" of the periodic covariance
    prior_sigma_0 = LogGaussian(mu=0, sigma=2 ** 2)  # offset in linear covariance

    # create map from kernel to priors
    prior_map = dict()

    prior_map['SE'] = {
        'variance': prior_sigma,
        'lengthscale': prior_l
    }

    prior_map['RQ'] = {
        'variance': prior_sigma,
        'lengthscale': prior_l,
        'power': prior_alpha
    }

    prior_map['PER'] = {
        'variance': prior_sigma,
        'lengthscale': prior_l,
        'period': prior_l_p
    }
    # make sure LinScaleCovariance is used
    prior_map['LIN'] = {
        'variances': prior_sigma,
        'shifts': prior_sigma_0
    }

    # likelihood hyperprior
    prior_map['GP'] = {
        'variance': prior_sigma_n
    }

    return prior_map

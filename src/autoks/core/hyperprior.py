from typing import Dict, Optional

import numpy as np

from src.autoks.core.prior import PriorDist
from src.evalg.serialization import Serializable

PriorMap = Dict[str, PriorDist]
PriorsMap = Dict[str, PriorMap]


class HyperpriorMap(Serializable):

    def __init__(self, prior_map: Optional[PriorsMap] = None):
        self.prior_map = prior_map or {}

    def to_dict(self):
        input_dict = super().to_dict()

        prior_map_cp = dict()
        for key, val in self.prior_map.items():
            prior_map_cp[key] = {}
            for param_name, prior in val.items():
                prior_map_cp[key][param_name] = prior.to_dict()

        input_dict["prior_map"] = prior_map_cp
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        if 'prior_map' in input_dict:
            for key, val in input_dict["prior_map"].items():
                for param_name, prior in val.items():
                    input_dict["prior_map"][key][param_name] = PriorDist.from_dict(prior)
        return input_dict

    def __len__(self):
        return len(self.prior_map)

    def __contains__(self, item):
        return item in self.prior_map

    def __getitem__(self, item):
        return self.prior_map[item]

    def __setitem__(self, key, value):
        self.prior_map[key] = value

    def __delitem__(self, key):
        del self.prior_map[key]


def boms_hyperpriors(data_noise: float = 0.01) -> HyperpriorMap:
    """

    Malkomes et al., 2016

    :return: The hyperparameter priors used in BOMS.
    """
    # length scales (period of a periodic covariance, etc.)
    prior_l = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(0.1), 'sigma': 1})

    # signal variance
    prior_sigma = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(0.4), 'sigma': 1})

    # observation noise
    prior_sigma_n = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(data_noise), 'sigma': np.sqrt(0.1)})

    # α parameter of the rational quadratic covariance
    prior_alpha = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(0.05), 'sigma': np.sqrt(0.5)})

    # the "length scale" of the periodic covariance
    prior_l_p = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(2), 'sigma': np.sqrt(0.5)})

    # offset in linear covariance
    prior_sigma_0 = PriorDist.from_prior_str("LOGNORMAL", {'mu': 0, 'sigma': 1})

    # period in standard periodic
    prior_p = PriorDist.from_prior_str("LOGNORMAL", {'mu': np.log(0.1), 'sigma': np.sqrt(0.5)})

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
        'lengthscale': prior_l_p,
        'period': prior_p
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

    return HyperpriorMap(prior_map)

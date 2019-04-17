import numpy as np

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

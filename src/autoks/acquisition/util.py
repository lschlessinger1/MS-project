import numpy as np
from scipy.special import erfc


def get_quantiles(acquisition_par: float,
                  f_max: float,
                  m: np.ndarray,
                  s: np.ndarray):
    """
    Quantiles of the Gaussian distribution useful to determine the acquisition function values.
    :param acquisition_par: parameter of the acquisition function
    :param f_max: current maximum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    """
    if isinstance(s, np.ndarray):
        s[s < 1e-10] = 1e-10
    elif s < 1e-10:
        s = 1e-10
    u = (m - f_max - acquisition_par) / s
    phi = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return phi, Phi, u

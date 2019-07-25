import numpy as np

from src.autoks.backend.model import RawGPModelType, n_data, log_likelihood
from src.autoks.model_selection_criteria import BIC


def log_likelihood_normalized(model: RawGPModelType) -> float:
    """Computes the normalized log likelihood.

    :param model:
    :return:
    """
    return log_likelihood(model) / n_data(model)


def likelihood_normalized(model: RawGPModelType) -> float:
    """Computes the normalized likelihood.

    :param model:
    :return:
    """
    return np.exp(log_likelihood(model)) / n_data(model)


def negative_bic(model: RawGPModelType) -> float:
    """Compute the negative of the BIC score."""
    return -BIC(model)

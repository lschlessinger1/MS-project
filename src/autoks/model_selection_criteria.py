from typing import List

import numpy as np

from src.autoks.backend.model import RawGPModelType, n_params, n_data, log_likelihood
from src.evalg.fitness import covariant_parsimony_pressure


def BIC(model: RawGPModelType) -> float:
    """Bayesian Information Criterion (BIC).

    Calculate the BIC for a Gaussian process model with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Bayesian_information_criterion

    BIC = ln(n)k - 2ln(L^)
    """
    n = n_data(model)
    k = n_params(model)
    return np.log(n) * k - 2 * log_likelihood(model)


def AIC(model: RawGPModelType) -> float:
    """Akaike Information Criterion (AIC).

    Calculate the AIC for a GPy `model` with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Akaike_information_criterion

    AIC = 2k - 2ln(L^)
    """
    k = n_params(model)
    return 2 * k - 2 * log_likelihood(model)


def pl2(model: RawGPModelType) -> float:
    """Compute the modified expected log-predictive likelihood (PL2) score of a model.

    Ando & Tsay, 2009
    :param model:
    :return:
    """
    n = n_data(model)
    k = n_params(model)
    nll = -log_likelihood(model)
    return nll / n + k / (2 * n)


def cov_parsimony_pressure(model: RawGPModelType,
                           model_scores: List[float],
                           model_sizes: List[int]) -> float:
    """Covariant parsimony pressure method of a model."""
    model_size = n_params(model)
    lml = log_likelihood(model)
    return covariant_parsimony_pressure(fitness=lml, size=model_size, sizes=model_sizes, fitness_list=model_scores)


# Model comparison scores

def bayes_factor(model_1: RawGPModelType,
                 model_2: RawGPModelType) -> float:
    """Compute the Bayes factor between two models.
    https://en.wikipedia.org/wiki/Bayes_factor

    :param model_1:
    :param model_2:
    :return:
    """
    model_evidence_1 = np.exp(log_likelihood(model_1))
    model_evidence_2 = np.exp(log_likelihood(model_2))
    return model_evidence_1 / model_evidence_2

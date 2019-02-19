import numpy as np
from GPy.core import GP
from GPy.kern import Kern

from autoks.kernel import kernel_to_infix_tokens, tokens_to_str
from evalg.encoding import infix_tokens_to_postfix_tokens, postfix_tokens_to_binexp_tree


def model_to_infix_tokens(model: GP):
    return kernel_to_infix_tokens(model.kern)


def model_to_infix(model: GP):
    infix_tokens = model_to_infix_tokens(model)
    return tokens_to_str(infix_tokens)


def model_to_binexptree(model: GP):
    infix_tokens = model_to_infix_tokens(model)
    postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
    tree = postfix_tokens_to_binexp_tree(postfix_tokens)
    return tree


def set_model_kern(model: GP, new_kern: Kern):
    model.unlink_parameter(model.kern)
    model.link_parameter(new_kern)
    model.kern = new_kern


def is_nan_model(model: GP):
    return np.isnan(model.param_array).any()


# Model selection criteria

def log_likelihood_normalized(model: GP):
    """Computes the normalized log likelihood."""
    dataset_size = model.X.shape[0]
    return model.log_likelihood() / dataset_size


def BIC(model: GP):
    """
    Calculate the Bayesian Information Criterion (BIC) for a GPy `model` with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    # model.log_likelihood() is the natural logarithm of the marginal likelihood of the Gaussian process.
    # len(model.X) is the number of data points.
    # model._size_transformed() is the number of optimisation parameters.
    # BIC = ln(n)k - 2ln(L^)
    n = len(model.X)
    k = model._size_transformed()
    return np.log(n) * k - 2 * model.log_likelihood()


def AIC(model: GP):
    """
    Calculate the Akaike Information Criterion (AIC) for a GPy `model` with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    # model.log_likelihood() is the natural logarithm of the marginal likelihood of the Gaussian process.
    # model._size_transformed() is the number of optimisation parameters.
    # AIC = 2k - 2ln(L^)
    k = model._size_transformed()
    return 2 * k - 2 * model.log_likelihood()


def pl2(model: GP):
    """ Compute the modified expected log-predictive likelihood (PL2) score of a model.

    Ando & Tsay, 2009
    :param model:
    :return:
    """
    n = len(model.X)
    k = model._size_transformed()
    nll = -model.log_likelihood()
    return nll / n + k / (2 * n)


# Model comparison scores

def bayes_factor(model_1: GP, model_2: GP):
    """ Compute the Bayes factor between two models
    https://en.wikipedia.org/wiki/Bayes_factor

    :param model_1:
    :param model_2:
    :return:
    """
    model_evidence_1 = np.exp(model_1.log_likelihood())
    model_evidence_2 = np.exp(model_2.log_likelihood())
    return model_evidence_1 / model_evidence_2

from typing import List, Optional, Tuple

import numpy as np
from GPy.core import GP
from GPy.core.model import Model

from src.autoks.backend.kernel import kernel_to_infix_tokens, tokens_to_str, RawKernelType
from src.evalg.encoding import infix_tokens_to_postfix_tokens, postfix_tokens_to_binexp_tree, BinaryTree
from src.evalg.fitness import covariant_parsimony_pressure

RawGPModelType = GP
RawModelType = Model


def model_to_infix_tokens(model: RawGPModelType) -> List[str]:
    """Convert a model to list of infix tokens.

    :param model:
    :return:
    """
    return kernel_to_infix_tokens(model.kern)


def model_to_infix(model: RawGPModelType) -> str:
    """Convert a model to an infix string

    :param model:
    :return:
    """
    infix_tokens = model_to_infix_tokens(model)
    return tokens_to_str(infix_tokens)


def model_to_binexptree(model: RawGPModelType) -> BinaryTree:
    """Convert a model to a binary expression tree.

    :param model:
    :return:
    """
    infix_tokens = model_to_infix_tokens(model)
    postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
    tree = postfix_tokens_to_binexp_tree(postfix_tokens)
    return tree


def save_model(m: RawModelType,
               output_file_name: str,
               compress: bool = False,
               save_data: bool = False) -> None:
    """Save a GPy model.

    :param m: The GPy model to save.
    :param output_file_name: The path of the file to save.
    :param compress: if true, save a ZIp, otherwise save a JSON file.
    :param save_data: A flag indicating whether or not to save the data.
    :return:
    """
    # GPy.model.Model.save_model is broken! Must use hidden function _save_model for now...
    # noinspection PyProtectedMember
    m._save_model(output_file_name, compress=compress, save_data=save_data)


def load_model(output_file_name: str,
               data: Optional[Tuple[np.ndarray, np.ndarray]]) -> RawModelType:
    """Load a GPy model.

    :param output_file_name: The path of the file containing the saved model.
    :param data: A tuple containing X and Y arrays. data = (x, y).
    :return: The GPy model.
    """
    return Model.load_model(output_file_name, data=data)


def set_model_kern(model: RawGPModelType,
                   new_kern: RawKernelType) -> None:
    """Set the kernel of a model.

    :param model:
    :param new_kern:
    :return:
    """
    model.unlink_parameter(model.kern)
    model.link_parameter(new_kern)
    model.kern = new_kern


def is_nan_model(model: RawGPModelType) -> bool:
    """Is a NaN model.

    :param model:
    :return:
    """
    return np.isnan(model.param_array).any()


# Model selection criteria

def log_likelihood_normalized(model: RawGPModelType) -> float:
    """Computes the normalized log likelihood.

    :param model:
    :return:
    """
    dataset_size = model.X.shape[0]
    return model.log_likelihood() / dataset_size


def BIC(model: RawGPModelType) -> float:
    """Bayesian Information Criterion (BIC).

    Calculate the BIC for a GPy `model` with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Bayesian_information_criterion

    BIC = ln(n)k - 2ln(L^)
    """
    # model.log_likelihood() is the natural logarithm of the marginal likelihood of the Gaussian process.
    # len(model.x) is the number of data points.
    # model._size_transformed() is the number of optimisation parameters.
    n = len(model.X)
    # noinspection PyProtectedMember
    k = model._size_transformed()
    return np.log(n) * k - 2 * model.log_likelihood()


def AIC(model: RawGPModelType) -> float:
    """Akaike Information Criterion (AIC).

    Calculate the AIC for a GPy `model` with maximum likelihood hyperparameters on a
    given dataset.
    https://en.wikipedia.org/wiki/Akaike_information_criterion

    AIC = 2k - 2ln(L^)
    """
    # model.log_likelihood() is the natural logarithm of the marginal likelihood of the Gaussian process.
    # model._size_transformed() is the number of optimisation parameters.
    # noinspection PyProtectedMember
    k = model._size_transformed()
    return 2 * k - 2 * model.log_likelihood()


def pl2(model: RawGPModelType) -> float:
    """Compute the modified expected log-predictive likelihood (PL2) score of a model.

    Ando & Tsay, 2009
    :param model:
    :return:
    """
    n = len(model.X)
    # noinspection PyProtectedMember
    k = model._size_transformed()
    nll = -model.log_likelihood()
    return nll / n + k / (2 * n)


def cov_parsimony_pressure(model: RawGPModelType,
                           model_scores: List[float],
                           model_sizes: List[int]) -> float:
    """Covariant parsimony pressure method of a model."""
    # noinspection PyProtectedMember
    model_size = model._size_transformed()
    lml = model.log_likelihood()
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
    model_evidence_1 = np.exp(model_1.log_likelihood())
    model_evidence_2 = np.exp(model_2.log_likelihood())
    return model_evidence_1 / model_evidence_2

from typing import List, Optional, Tuple

import numpy as np
from GPy.core import GP
from GPy.core.model import Model

from src.autoks.backend.kernel import kernel_to_infix_tokens, tokens_to_str, RawKernelType
from src.evalg.encoding import infix_tokens_to_postfix_tokens, postfix_tokens_to_binexp_tree, BinaryTree

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


def n_data(model: RawGPModelType) -> int:
    """Get the number of data points of a model."""
    return model.num_data


def n_params(model: RawGPModelType) -> int:
    """Get the number of optimization parameters."""
    # noinspection PyProtectedMember
    return model._size_transformed()


def log_likelihood(model: RawGPModelType) -> float:
    """Get the natural logarithm of the marginal likelihood of the Gaussian process model."""
    return model.log_likelihood()

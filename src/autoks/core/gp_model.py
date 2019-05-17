from typing import Optional, List, FrozenSet

import numpy as np
from GPy.core import GP
from GPy.likelihoods import Likelihood

from src.autoks.backend.kernel import encode_prior, get_priors, encode_kernel, sort_kernel, kernel_to_infix
from src.autoks.core.covariance import Covariance, pretty_print_covariances
from src.autoks.util import remove_duplicates


class GPModel:
    """GPy kernel wrapper
    """
    covariance: Covariance
    lik_params: Optional[np.ndarray]
    evaluated: bool
    nan_scored: bool
    expanded: bool
    model_input_dict: Optional[dict]
    likelihood: Optional[Likelihood]

    def __init__(self, covariance: Covariance, likelihood=None, evaluated=False, nan_scored=False, expanded=False):
        self.covariance = covariance
        self.evaluated = evaluated
        self.nan_scored = nan_scored
        self.expanded = expanded
        self._score = None

        self.model_input_dict = dict()  # Only save keyword arguments of GP
        self.likelihood = likelihood

    def score_model(self, x, y, scoring_func):
        model = self.fit(x, y)
        self.score = scoring_func(model)
        return self.score

    def build_model(self, x, y) -> GP:
        input_dict = self.model_input_dict.copy()

        input_dict['X'] = x
        input_dict['Y'] = y
        input_dict['kernel'] = self.covariance.raw_kernel
        input_dict['likelihood'] = self.likelihood

        gp = GP(**input_dict)

        return gp

    def fit(self, x, y, optimizer=None, n_restarts=10):
        model = self.build_model(x, y)
        model.optimize_restarts(ipython_notebook=False, optimizer=optimizer, num_restarts=n_restarts, verbose=False,
                                robust=True, messages=False)
        return model

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score
        # Update evaluated as well
        self.evaluated = True

    def __str__(self):
        return str(self.covariance.symbolic_expr)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'covariance={self.covariance!r}, evaluated={self.evaluated!r}, ' \
            f'nan_scored={self.nan_scored!r}, expanded={self.expanded!r}, score={self.score!r})'


def pretty_print_gp_models(gp_models: List[GPModel],
                           kernel_type_label: Optional[str] = None):
    covariances = [gp_model.covariance for gp_model in gp_models]
    pretty_print_covariances(covariances, kernel_type_label)


def remove_duplicate_gp_models(kernels: List[GPModel]) -> List[GPModel]:
    """Remove duplicate GPModel's.

    prioritizing when removing duplicates
        1. highest score
        2. not nan scored
        3. evaluated

    :param kernels:
    :return:
    """

    unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated and not kernel.nan_scored]
    evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
    nan_scored_kernels = [kernel for kernel in kernels if kernel.nan_scored]
    # Prioritize highest scoring kernel for duplicates
    sorted_evaluated_kernels = sorted(evaluated_kernels, key=lambda k: k.score, reverse=True)

    # Assume precedence by order.
    gp_models = sorted_evaluated_kernels + nan_scored_kernels + unevaluated_kernels
    return remove_duplicates([gp_model.covariance.symbolic_expr_expanded for gp_model in gp_models], gp_models)


def all_same_expansion(new_kernels: List[GPModel],
                       prev_expansions: List[FrozenSet[str]],
                       max_expansions: int) -> bool:
    kernels_infix_new = model_to_infix_set(new_kernels)
    all_same = all([s == kernels_infix_new for s in prev_expansions])
    return all_same and len(prev_expansions) == max_expansions


def model_to_infix_set(gp_models: List[GPModel]) -> FrozenSet[str]:
    kernels_sorted = [sort_kernel(gp_model.covariance.raw_kernel) for gp_model in gp_models]
    return frozenset([kernel_to_infix(kernel) for kernel in kernels_sorted])


def update_kernel_infix_set(new_kernels: List[GPModel],
                            prev_expansions: List[FrozenSet[str]],
                            max_expansions: int) -> List[FrozenSet[str]]:
    expansions = prev_expansions.copy()
    if len(prev_expansions) == max_expansions:
        expansions = expansions[1:]
    elif len(prev_expansions) < max_expansions:
        expansions += [model_to_infix_set(new_kernels)]

    return expansions


def randomize_models(gp_models: List[GPModel]) -> List[GPModel]:
    for gp_model in gp_models:
        gp_model.covariance.raw_kernel.randomize()

    return gp_models


def remove_nan_scored_models(gp_models: List[GPModel]) -> List[GPModel]:
    """Remove all models that have NaN scores.

    :param gp_models:
    :return:
    """
    return [gp_model for gp_model in gp_models if not gp_model.nan_scored]


def encode_gp_model(gp_model: GPModel) -> List[str]:
    """Encode GPModel

    :param gp_model:
    :return:
    """
    try:
        prior_enc = [encode_prior(prior) for prior in get_priors(gp_model.covariance.raw_kernel)]
    except ValueError:
        prior_enc = None
    return [encode_kernel(gp_model.covariance.raw_kernel), [prior_enc]]


def encode_gp_models(gp_models: List[GPModel]) -> np.ndarray:
    """Encode a list of models.

    :param gp_models: the models to encode.
    :return: An array containing encodings of the gp_models.
    """
    enc = np.empty((len(gp_models), 1), dtype=np.object)
    for i, gp_model in enumerate(gp_models):
        enc[i, 0] = encode_gp_model(gp_model)
    return enc

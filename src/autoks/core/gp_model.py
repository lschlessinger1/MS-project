import logging
from typing import Optional, List, FrozenSet, Callable

import numpy as np
from GPy.likelihoods import Likelihood

from src.autoks.backend.kernel import encode_prior, get_priors, encode_kernel, sort_kernel, kernel_to_infix
from src.autoks.backend.model import RawGPModelType
from src.autoks.core.covariance import Covariance, pretty_print_covariances
from src.autoks.util import remove_duplicates
from src.evalg.serialization import Serializable


class GPModel(Serializable):
    """GPy model wrapper."""
    model_input_dict: Optional[dict]

    def __init__(self,
                 covariance: Covariance,
                 likelihood: Optional[Likelihood] = None,
                 evaluated: bool = False):
        self.covariance = covariance
        self.evaluated = evaluated
        self._score = np.nan

        self.model_input_dict = dict()  # Only save keyword arguments of GP
        self.likelihood = likelihood

        self.failed_fitting = False

    def score_model(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    scoring_func: Callable[[RawGPModelType], float],
                    **fit_kwargs):
        model = self.fit(x, y, **fit_kwargs)

        if not self.failed_fitting:
            self.score = scoring_func(model)
        else:
            self.score = np.nan
        return self.score

    def build_model(self,
                    x: np.ndarray,
                    y: np.ndarray) -> RawGPModelType:
        input_dict = self.model_input_dict.copy()

        input_dict['X'] = x
        input_dict['Y'] = y
        input_dict['kernel'] = self.covariance.raw_kernel
        input_dict['likelihood'] = self.likelihood

        gp = RawGPModelType(**input_dict)

        return gp

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            optimizer: str = None,
            n_restarts: int = 10) -> RawGPModelType:
        model = self.build_model(x, y)
        model.optimize_restarts(ipython_notebook=False, optimizer=optimizer, num_restarts=n_restarts, verbose=False,
                                robust=True, messages=False)

        # Test that optimization was successful.
        bad_values = (float('-inf'), float('inf'), np.nan)
        if model.log_likelihood() in bad_values:
            logging.warning(f'Model log likelihood is {model.log_likelihood()}. Attempting to restart optimization.')
            # Try to recover from numerical issue by restarting optimization runs.
            model.optimize_restarts(ipython_notebook=False, optimizer=optimizer, num_restarts=n_restarts, verbose=False,
                                    robust=True, messages=False)
            # If still bad value, flag this model.
            if model.log_likelihood() in bad_values:
                logging.warning(f'Model failed fitting (again). log likelihood is {model.log_likelihood()}.')
                self.failed_fitting = True

        self.covariance.raw_kernel = model.kern
        self.likelihood = model.likelihood

        return model

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score
        # Update evaluated as well
        self.evaluated = True

    def to_dict(self) -> dict:
        # Get likelihood and covariance
        input_dict = super().to_dict()

        input_dict["likelihood"] = None if self.likelihood is None else self.likelihood.to_dict()
        input_dict["covariance"] = self.covariance.to_dict()
        input_dict["score"] = self.score

        return input_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        score = input_dict.pop('score')
        gp_model = super()._build_from_input_dict(input_dict)
        gp_model.score = score
        return gp_model

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        input_dict["covariance"] = Covariance.from_dict(input_dict["covariance"])
        input_dict["likelihood"] = None if input_dict['likelihood'] is None else \
            Likelihood.from_dict(input_dict["likelihood"])
        return input_dict

    def __str__(self):
        return str(self.covariance.symbolic_expr)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'covariance={self.covariance!r}, evaluated={self.evaluated!r}, ' \
            f'score={self.score!r})'


def pretty_print_gp_models(gp_models: List[GPModel],
                           kernel_type_label: Optional[str] = None):
    covariances = [gp_model.covariance for gp_model in gp_models]
    pretty_print_covariances(covariances, kernel_type_label)


def remove_duplicate_gp_models(kernels: List[GPModel], verbose: bool = False) -> List[GPModel]:
    """Remove duplicate GPModel's.

    prioritizing when removing duplicates
        1. highest score
        2. evaluated

    :param kernels:
    :param verbose:
    :return:
    """
    unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
    evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
    # Prioritize highest scoring kernel for duplicates
    sorted_evaluated_kernels = sorted(evaluated_kernels, key=lambda k: k.score, reverse=True)

    # Assume precedence by order.
    gp_models = sorted_evaluated_kernels + unevaluated_kernels
    deduped = remove_duplicates([gp_model.covariance.symbolic_expr_expanded for gp_model in gp_models], gp_models)

    if verbose:
        n_before = len(gp_models)
        n_removed = n_before - len(deduped)
        print(f'Removed {n_removed} duplicate GP models.\n')

    return deduped


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

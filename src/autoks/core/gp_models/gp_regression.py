from typing import Optional

from GPy import likelihoods
from GPy.inference.latent_function_inference import Laplace

from src.autoks.backend.kernel import set_priors
from src.autoks.core.hyperprior import PriorMap


def gp_regression(inference_method: Optional[str] = None,
                  likelihood_hyperprior: Optional[PriorMap] = None) -> dict:
    """Build model dict of GP regression."""
    model_dict = dict()

    # Set likelihood.
    likelihood = likelihoods.Gaussian()
    if likelihood_hyperprior is not None:
        # set likelihood hyperpriors
        likelihood = set_priors(likelihood, likelihood_hyperprior)
    model_dict['likelihood'] = likelihood

    # Set inference method.
    if inference_method == 'laplace':
        inference_method = Laplace()
    model_dict['inference_method'] = inference_method

    return model_dict

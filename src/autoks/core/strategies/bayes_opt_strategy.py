from typing import Callable

import numpy as np
from GPy.kern import RBFDistanceBuilderKernelKernel

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import BomsGrammar
from src.autoks.core.hyperprior import HyperpriorMap
from src.autoks.debugging import assert_valid_kernel_kernel
from src.autoks.distance.distance import ActiveModels, DistanceBuilder
from src.autoks.gp_regression_models import KernelKernelGPModel


class BayesOptStrategy:
    """Bayesian optimization strategy"""

    def __init__(self,
                 active_models: ActiveModels,
                 acquisition_fn: Callable,
                 kernel_builder: DistanceBuilder,
                 kernel_kernel_hyperpriors: HyperpriorMap):
        self.active_models = active_models
        self.acquisition_fn = acquisition_fn
        self.kernel_builder = kernel_builder

        kernel_kernel = Covariance(
            RBFDistanceBuilderKernelKernel(self.kernel_builder, n_models=len(self.active_models)))
        self.kernel_kernel_gp_model = KernelKernelGPModel(kernel_kernel, verbose=False, exact_f_eval=False,
                                                          kernel_kernel_hyperpriors=kernel_kernel_hyperpriors,
                                                          optimize_restarts=10)

    def query(self, fitness_scores, x_train, y_train, eval_budget, gp_fn, gp_args,
              **ms_args):
        """"""
        model_proposer = BomsGrammar()
        model_proposer.build(x_train.shape[1])
        candidate_covariances = model_proposer.expand(seed_models=self.active_models.get_selected_models())
        likelihood = self.active_models.get_selected_models()[0].likelihood
        candidate_models = [GPModel(cov, likelihood) for cov in candidate_covariances]

        # Update active models.
        new_candidate_indices = self.active_models.update(candidate_models)

        # Pool of models.
        all_candidate_indices = self.active_models.get_candidate_indices()
        selected_indices = self.active_models.selected_indices

        # Update model distances using the kernel builder.
        self.kernel_builder.update(self.active_models, new_candidate_indices,
                                   all_candidate_indices,
                                   selected_indices, x_train)

        # Make sure all necessary indices are not NaN.
        assert_valid_kernel_kernel(self.kernel_builder, len(self.active_models), selected_indices,
                                   all_candidate_indices)

        meta_x_train = np.array(selected_indices)[:, None]
        meta_y_train = np.array(fitness_scores)[:, None]

        # Train the GP.
        self.kernel_kernel_gp_model.update(meta_x_train, meta_y_train, None, None)
        # Housekeeping for kernel kernel. Must update the number of active models.
        self.kernel_kernel_gp_model.model.kern.n_models = len(self.active_models)

        # Compute acquisition function values.
        x_test = np.array(all_candidate_indices)[:, None]
        acq_scores = self.acquisition_fn(x_test, self.kernel_kernel_gp_model).flatten().tolist()

        indices_acquisition = np.argsort(np.array(acq_scores).flatten())

        # Argmax acquisition function.
        next_model_index = all_candidate_indices[indices_acquisition[-1]]
        next_node = self.active_models.models[next_model_index]
        next_model = GPModel(next_node.covariance, next_node.likelihood)

        # Save next model index.
        self.active_models.selected_indices += [next_model_index]

        # Set remove priority.
        self.active_models.remove_priority = [all_candidate_indices[i] for i in indices_acquisition]

        return next_model

from typing import Callable

import numpy as np
from GPy.kern import RBFDistanceBuilderKernelKernel

from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.hyperprior import HyperpriorMap
from src.autoks.core.model_selection.evolutionary_model_selector import BoemsSurrogateSelector
from src.autoks.debugging import assert_valid_kernel_kernel
from src.autoks.distance.distance import DistanceBuilder, ActiveModels
from src.autoks.gp_regression_models import KernelKernelGPModel


class BayesOptGPStrategy:
    """Bayesian optimization genetic programming strategy"""

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
        self.kernel_kernel_gp_model_train_freq = 3
        self.expansion_freq = 3

    def query(self, selected_models, fitness_scores, x_train, y_train, eval_budget, gp_fn, gp_args,
              **ms_args) -> GPModel:
        """Get next model.

        :param selected_models:
        :param fitness_scores:
        :param candidate_models:
        :return:
        """
        generation = len(selected_models)

        meta_x_train = np.array(self.active_models.selected_indices)[:, None]
        meta_y_train = np.array(fitness_scores)[:, None]

        # Check shapes of meta_x and meta_y
        assert meta_x_train.ndim == 2 and meta_y_train.ndim == 2
        assert meta_x_train.shape == (len(selected_models), 1) and meta_y_train.shape == (len(fitness_scores), 1)
        assert meta_x_train.shape == meta_y_train.shape

        self.kernel_kernel_gp_model.update(meta_x_train, meta_y_train, None, None)

        # Compute new evaluated models vs all old candidates.
        newly_selected_ind = [self.active_models.selected_indices[-1]]
        old_candidates = self.active_models.get_candidate_indices()
        self.kernel_builder.compute_distance(self.active_models, newly_selected_ind, old_candidates)

        # Make sure all necessary indices are not NaN.
        assert_valid_kernel_kernel(self.kernel_builder, len(self.active_models), self.active_models.selected_indices,
                                   self.active_models.get_candidate_indices())

        # Compute acquisition scores for all old candidates
        # Pool of models.
        selected_indices = self.active_models.selected_indices
        all_candidate_indices = set(range(len(self.active_models)))
        all_candidate_indices = list(all_candidate_indices - set(selected_indices))

        # Train the GP.
        self.kernel_kernel_gp_model.model.kern.n_models = len(self.active_models)
        if generation % self.kernel_kernel_gp_model_train_freq == 0:
            self.kernel_kernel_gp_model.train()

        x_test = np.array(all_candidate_indices)[:, None]
        acq_scores = self.acquisition_fn(x_test, self.kernel_kernel_gp_model).tolist()
        assert_valid_kernel_kernel(self.kernel_builder, len(self.active_models), self.active_models.selected_indices,
                                   self.active_models.get_candidate_indices())

        assert not np.isnan(acq_scores).any()
        for i, score in zip(all_candidate_indices, acq_scores):
            self.active_models.models[i].score = score[0]

        # Now search for higher acquisition scores using genetic programming.
        if generation == 1 or generation % self.expansion_freq == 0:
            model_selector = BoemsSurrogateSelector(self.active_models, self.acquisition_fn,
                                                    self.kernel_kernel_gp_model, gp_fn=gp_fn, gp_args=gp_args,
                                                    **ms_args)
            model_selector.train(x_train, y_train, eval_budget=eval_budget, verbose=0)

        # Compute acquisition scores for all old candidates
        # Pool of models.
        all_candidate_indices = set(range(len(self.active_models)))
        all_candidate_indices = list(all_candidate_indices - set(selected_indices))
        acq_scores = [self.active_models.models[i].score for i in all_candidate_indices]
        next_model_index = all_candidate_indices[int(np.nanargmax([acq_scores]))]
        next_model = self.active_models.models[next_model_index]

        # Reinitialize to get rid of acquisition score.
        next_model = GPModel(next_model.covariance, next_model.likelihood)

        # think about how to handle model in population/selected model that is not in active models.
        # how can this happen?
        # save next model index
        try:
            next_selected_index = self.active_models.index(next_model)
        except KeyError:
            next_selected_index = self.active_models.update([next_model])[0]
            print('Inserting next selected model into active models')
            self.kernel_builder.precompute_information(self.active_models, [next_selected_index], x_train)
        self.active_models.selected_indices += [next_selected_index]

        # check that scores of previously selected models are not 0
        return next_model

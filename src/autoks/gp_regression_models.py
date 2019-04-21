from typing import List

import numpy as np
from GPy import likelihoods
from GPy.core import GP

from src.autoks.debugging import test_kernel
from src.autoks.distance.distance import ActiveModels


class KernelKernelGPRegression(GP):

    def __init__(self, X, Y, kernel, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):
        likelihood = likelihoods.Gaussian(variance=noise_var)

        super().__init__(X, Y, kernel, likelihood, name='GP model regression', Y_metadata=Y_metadata,
                         normalizer=normalizer, mean_function=mean_function)

    def update(self,
               active_models: ActiveModels,
               new_candidates_indices: List[int],
               all_candidates_indices: List[int],
               old_selected_indices: List[int],
               new_selected_indices: List[int],
               data_X: np.ndarray) -> None:
        """Update kernel and distance builder

        :param active_models: models including evaluated and non-evaluated
        :param new_candidates_indices: indices in active models
        :param all_candidates_indices: All in active model except
        :param new_selected_indices: newly evaluated indices
        :param old_selected_indices: previously evaluated indices
        :param data_X:
        :return:
        """
        self.kern.distance_builder.update_multiple(active_models, new_candidates_indices, all_candidates_indices,
                                                   old_selected_indices, new_selected_indices, data_X)
        self.kern.n_models = len(active_models)

        # for debugging
        selected = old_selected_indices + new_selected_indices
        test_kernel(self.kern.distance_builder, len(active_models), selected, all_candidates_indices)

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return KernelKernelGPRegression(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer,
                                        gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(KernelKernelGPRegression, self).to_dict(save_data)
        model_dict["class"] = "GPy.models.KernelKernelGPRegression"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return KernelKernelGPRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=compress, save_data=save_data)

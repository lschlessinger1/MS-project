from typing import List

import numpy as np
from GPy.inference.latent_function_inference import Laplace

from src.autoks.backend.model import log_likelihood_normalized
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.grammar import RandomGrammar
from src.autoks.core.model_selection.base import ModelSelector


class RandomModelSelector(ModelSelector):
    grammar: RandomGrammar

    def __init__(self, grammar, objective=None, n_parents: int = 1, additive_form=False, optimizer=None,
                 n_restarts_optimizer=3, use_laplace=True):
        if objective is None:
            objective = log_likelihood_normalized

        if use_laplace:
            inference_method = Laplace()
        else:
            inference_method = None

        likelihood = None

        super().__init__(grammar, objective, n_parents, additive_form, likelihood, inference_method, optimizer,
                         n_restarts_optimizer)

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        return self.grammar.base_kernels

    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               eval_budget: int,
               max_generations: int,
               verbose: int = 0) -> GPModelPopulation:
        population = self.initialize(x, y, eval_budget, verbose=verbose)
        self.active_set_callback(population.models, self, x, y)

        depth = 0
        while self.n_evals < eval_budget:
            if depth > max_generations:
                break

            self._print_search_summary(depth, population, eval_budget, max_generations, verbose=verbose)

            new_models = self.propose_new_models(population, verbose=verbose)
            self.expansion_callback(new_models, self, x, y)
            population.models = new_models

            self.evaluate_models(population.candidates(), x, y, eval_budget, verbose=verbose)

            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

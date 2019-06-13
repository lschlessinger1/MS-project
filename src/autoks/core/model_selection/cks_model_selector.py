from typing import List, Callable, Optional

import numpy as np

from src.autoks.backend.model import BIC
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.gp_models import gp_regression
from src.autoks.core.grammar import CKSGrammar
from src.autoks.core.model_selection.base import ModelSelector


class CKSModelSelector(ModelSelector):
    grammar: CKSGrammar

    def __init__(self, grammar, fitness_fn=None, n_parents: int = 1, additive_form=False,
                 gp_fn: Callable = gp_regression, gp_args: Optional[dict] = None, optimizer='scg',
                 n_restarts_optimizer=3):

        if fitness_fn is None:
            def negative_BIC(m):
                """Computes the negative of the Bayesian Information Criterion (BIC)."""
                return -BIC(m)

            # Use the negative BIC because we want to maximize the fitness_fn.
            fitness_fn = negative_BIC

        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer,
                         n_restarts_optimizer)

    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               eval_budget: int,
               max_generations: int = 10,
               verbose: int = 1) -> GPModelPopulation:
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

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        return self.grammar.base_kernels

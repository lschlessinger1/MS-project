from typing import List, Callable, Optional

import numpy as np

from src.autoks.backend.model import log_likelihood_normalized, RawGPModelType
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.gp_models import gp_regression
from src.autoks.core.grammar import RandomGrammar
from src.autoks.core.model_selection.base import ModelSelector


class RandomModelSelector(ModelSelector):

    def __init__(self,
                 grammar: Optional[RandomGrammar],
                 fitness_fn: Callable[[RawGPModelType], float] = log_likelihood_normalized,
                 n_parents: int = 1,
                 additive_form: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Callable = gp_regression,
                 gp_args: Optional[dict] = None):
        if grammar is None:
            grammar = RandomGrammar()

        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer, n_restarts_optimizer)

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

            new_models = self.propose_new_models(population, x, y, verbose=verbose)
            self.expansion_callback(new_models, self, x, y)
            population.models = new_models

            self.evaluate_models(population.candidates(), x, y, eval_budget, verbose=verbose)

            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

from typing import List, Callable, Optional

import numpy as np

from src.autoks.backend.model import log_likelihood_normalized
from src.autoks.core.acquisition_function import ExpectedImprovementPerSec
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.gp_models import gp_regression
from src.autoks.core.grammar import BOMSGrammar
from src.autoks.core.model_selection.base import SurrogateBasedModelSelector
from src.autoks.core.query_strategy import BestScoreStrategy


class BomsModelSelector(SurrogateBasedModelSelector):
    grammar: BOMSGrammar

    def __init__(self, grammar, fitness_fn=None, n_parents: int = 1, query_strategy=None, additive_form=False,
                 gp_fn: Callable = gp_regression, gp_args: Optional[dict] = None, optimizer=None,
                 n_restarts_optimizer=3):
        if fitness_fn is None:
            fitness_fn = log_likelihood_normalized

        if query_strategy is None:
            acq = ExpectedImprovementPerSec()
            query_strategy = BestScoreStrategy(scoring_func=acq)

        # Use laplace inference by default.
        if gp_args is not None and 'inference_method' not in gp_args:
            gp_args['inference_method'] = 'laplace'
        else:
            gp_args = {'inference_method': 'laplace'}

        super().__init__(grammar, fitness_fn, n_parents, query_strategy, additive_form, gp_fn, gp_args, optimizer,
                         n_restarts_optimizer)

    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               eval_budget: int,
               max_generations: int,
               verbose: int = 1) -> GPModelPopulation:
        pass

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        initial_level_depth = 2
        max_number_of_initial_models = 500
        initial_candidates = self.grammar.expand_full_brute_force(initial_level_depth, max_number_of_initial_models)
        return initial_candidates

    def initialize(self, x, y, eval_budget: int, verbose: int = 0) -> GPModelPopulation:
        population = GPModelPopulation()

        # initialize models
        initial_candidates = self.get_initial_candidates()
        indices = [0]
        initial_models = [initial_candidates[i] for i in indices]

        self.evaluate_models(initial_models, x, y, eval_budget, verbose=verbose)

        population.update(initial_candidates)

        return population

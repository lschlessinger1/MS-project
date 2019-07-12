from typing import List, Callable, Optional, Union

import numpy as np

from src.autoks.backend.model import RawGPModelType
from src.autoks.core.acquisition_function import ExpectedImprovementPerSec
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.grammar import BomsGrammar
from src.autoks.core.hyperprior import boms_hyperpriors
from src.autoks.core.model_selection.base import SurrogateBasedModelSelector
from src.autoks.core.query_strategy import BestScoreStrategy, QueryStrategy


class BomsModelSelector(SurrogateBasedModelSelector):

    def __init__(self,
                 grammar: Optional[BomsGrammar] = None,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 n_parents: int = 1,
                 query_strategy: Optional[QueryStrategy] = None,
                 additive_form: bool = False,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3):

        if grammar is None:
            hyperpriors = boms_hyperpriors()
            grammar = BomsGrammar(hyperpriors=hyperpriors)

        if query_strategy is None:
            acq = ExpectedImprovementPerSec()
            query_strategy = BestScoreStrategy(scoring_func=acq)

        # Use Laplace inference by default.
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

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        standardize_x = input_dict.pop('standardize_x')
        standardize_y = input_dict.pop('standardize_y')

        model_selector = super()._build_from_input_dict(input_dict)

        model_selector.standardize_x = standardize_x
        model_selector.standardize_y = standardize_y

        return model_selector

    def __str__(self):
        return self.name

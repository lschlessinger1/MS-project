from typing import List

from src.autoks.backend.model import log_likelihood_normalized
from src.autoks.core.acquisition_function import ExpectedImprovementPerSec
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.grammar import BOMSGrammar
from src.autoks.core.model_selection.model_selector import SurrogateBasedModelSelector
from src.autoks.core.query_strategy import BestScoreStrategy


class BomsModelSelector(SurrogateBasedModelSelector):
    grammar: BOMSGrammar

    def __init__(self, grammar, objective=None, eval_budget=50, max_generations=None, n_parents: int = 1,
                 query_strategy=None, additive_form=False, debug=False, verbose=False, optimizer=None,
                 n_restarts_optimizer=10, use_laplace=True, active_set_callback=None, eval_callback=None,
                 expansion_callback=None):
        if objective is None:
            objective = log_likelihood_normalized

        if query_strategy is None:
            acq = ExpectedImprovementPerSec()
            query_strategy = BestScoreStrategy(scoring_func=acq)

        super().__init__(grammar, objective, eval_budget, max_generations, n_parents, query_strategy,
                         additive_form, debug, verbose, optimizer, n_restarts_optimizer, use_laplace,
                         active_set_callback, eval_callback, expansion_callback)

    def _train(self, x, y) -> GPModelPopulation:
        pass

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        initial_level_depth = 2
        max_number_of_initial_models = 500
        initial_candidates = self.grammar.expand_full_brute_force(initial_level_depth, max_number_of_initial_models)
        return initial_candidates

    def initialize(self, x, y) -> GPModelPopulation:
        population = GPModelPopulation()

        # initialize models
        initial_candidates = self.get_initial_candidates()
        indices = [0]
        initial_models = [initial_candidates[i] for i in indices]

        self.evaluate_models(initial_models, x, y)

        population.update(initial_candidates)

        return population

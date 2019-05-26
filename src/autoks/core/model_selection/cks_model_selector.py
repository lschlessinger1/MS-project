from typing import List

from src.autoks.backend.model import BIC
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.grammar import CKSGrammar
from src.autoks.core.model_selection.model_selector import ModelSelector


class CKSModelSelector(ModelSelector):
    grammar: CKSGrammar

    def __init__(self, grammar, objective=None, eval_budget=50, max_generations=10, n_parents: int = 1,
                 additive_form=False, debug=False, verbose=False, optimizer='scg', n_restarts_optimizer=10,
                 use_laplace=True, active_set_callback=None, eval_callback=None, expansion_callback=None):

        if objective is None:
            def negative_BIC(m):
                """Computes the negative of the Bayesian Information Criterion (BIC)."""
                return -BIC(m)

            # Use the negative BIC because we want to maximize the objective.
            objective = negative_BIC

        super().__init__(grammar, objective, eval_budget, max_generations, n_parents, additive_form, debug,
                         verbose, optimizer, n_restarts_optimizer, use_laplace, active_set_callback, eval_callback,
                         expansion_callback)

    def _train(self, x, y) -> GPModelPopulation:
        population = self.initialize(x, y)
        self.active_set_callback(population.models, self, x, y)

        depth = 0
        while self.n_evals < self.eval_budget:
            if depth > self.max_generations:
                break

            self._print_search_summary(depth, population)

            new_models = self.propose_new_models(population)
            self.expansion_callback(new_models, self, x, y)
            population.models = new_models

            self.evaluate_models(population.candidates(), x, y)

            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        return self.grammar.base_kernels

from typing import List

import numpy as np

from src.autoks.backend.model import log_likelihood_normalized
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import EvolutionaryGrammar
from src.autoks.core.kernel_encoding import tree_to_kernel
from src.autoks.core.model_selection.model_selector import ModelSelector, SurrogateBasedModelSelector
from src.evalg.selection import ExponentialRankingSelector, TruncationSelector


class EvolutionaryModelSelector(ModelSelector):
    grammar: EvolutionaryGrammar

    def __init__(self, grammar, objective=None, initializer=None, n_init_trees=10, eval_budget=50, max_generations=None,
                 n_parents=10, max_offspring: int = 25, additive_form=False, debug=False, verbose=False, optimizer=None,
                 n_restarts_optimizer=10, use_laplace=True, active_set_callback=None, eval_callback=None,
                 expansion_callback=None):
        if objective is None:
            objective = log_likelihood_normalized

        super().__init__(grammar, objective, eval_budget, max_generations, n_parents, additive_form, debug, verbose,
                         optimizer, n_restarts_optimizer, use_laplace, active_set_callback, eval_callback,
                         expansion_callback)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = max_offspring

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
            population.update(new_models)

            self.evaluate_models(population.candidates(), x, y)

            population.models = self.select_offspring(population)
            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

    def select_parents(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select parents to expand.

        Here, exponential ranking selection is used.
        """
        selector = ExponentialRankingSelector(self.n_parents, c=0.7)
        return list(selector.select(np.array(population.models), np.array(population.objectives())).tolist())

    def select_offspring(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select offspring for the next generation.

        Here, the top k offspring are chosen to seed the next generation.
        """
        selector = TruncationSelector(self.max_offspring)
        offspring = list(selector.select(np.array(population.models), np.array(population.objectives())).tolist())
        return offspring

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        if self.initializer is not None:
            # Generate trees
            trees = [self.initializer.generate() for _ in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            covariances = [Covariance(k) for k in kernels]

            return covariances
        else:
            return self.grammar.base_kernels


class SurrogateEvolutionaryModelSelector(SurrogateBasedModelSelector):
    grammar: EvolutionaryGrammar

    def __init__(self, grammar, objective=None, query_strategy=None, initializer=None, n_init_trees=10, eval_budget=50,
                 max_generations=None, n_parents=10, max_offspring: int = 25, additive_form=False, debug=False,
                 verbose=False, optimizer=None, n_restarts_optimizer=10, use_laplace=True, active_set_callback=None,
                 eval_callback=None, expansion_callback=None):

        if objective is None:
            objective = log_likelihood_normalized

        super().__init__(grammar, objective, query_strategy, eval_budget, max_generations, n_parents, additive_form,
                         debug, verbose, optimizer, n_restarts_optimizer, use_laplace, active_set_callback,
                         eval_callback, expansion_callback)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = max_offspring

    def _train(self, x, y) -> GPModelPopulation:
        pass

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        if self.initializer is not None:
            # Generate trees
            trees = [self.initializer.generate() for _ in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            covariances = [Covariance(k) for k in kernels]

            return covariances
        else:
            return self.grammar.base_kernels

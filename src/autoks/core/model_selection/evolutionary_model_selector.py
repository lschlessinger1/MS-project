from typing import List, Callable, Optional

import numpy as np

from src.autoks.backend.model import log_likelihood_normalized
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.gp_models import gp_regression
from src.autoks.core.grammar import EvolutionaryGrammar
from src.autoks.core.kernel_encoding import tree_to_kernel
from src.autoks.core.model_selection.base import ModelSelector, SurrogateBasedModelSelector
from src.evalg.selection import ExponentialRankingSelector, TruncationSelector


class EvolutionaryModelSelector(ModelSelector):
    grammar: EvolutionaryGrammar

    def __init__(self, grammar, fitness_fn=None, initializer=None, n_init_trees=10, n_parents=10,
                 max_offspring: int = 25, additive_form=False, optimizer=None, n_restarts_optimizer=3,
                 gp_fn: Callable = gp_regression, gp_args: Optional[dict] = None):
        if fitness_fn is None:
            fitness_fn = log_likelihood_normalized

        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer,
                         n_restarts_optimizer)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = max_offspring

    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               eval_budget: int,
               max_generations: int,
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
            population.update(new_models)

            self.evaluate_models(population.candidates(), x, y, eval_budget, verbose=verbose)

            population.models = self.select_offspring(population)
            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

    def select_parents(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select parents to expand.

        Here, exponential ranking selection is used.
        """
        selector = ExponentialRankingSelector(self.n_parents, c=0.7)
        return list(selector.select(np.array(population.models), np.array(population.fitness_scores())).tolist())

    def select_offspring(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select offspring for the next generation.

        Here, the top k offspring are chosen to seed the next generation.
        """
        selector = TruncationSelector(self.max_offspring)
        offspring = list(selector.select(np.array(population.models), np.array(population.fitness_scores())).tolist())
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

    def __init__(self, grammar, fitness_fn=None, query_strategy=None, initializer=None, n_init_trees=10, n_parents=10,
                 max_offspring: int = 25, additive_form=False, optimizer=None, n_restarts_optimizer=10,
                 gp_fn: Callable = gp_regression, gp_args: Optional[dict] = None):

        if fitness_fn is None:
            fitness_fn = log_likelihood_normalized

        super().__init__(grammar, fitness_fn, query_strategy, n_parents, additive_form, gp_fn, gp_args, optimizer,
                         n_restarts_optimizer)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = max_offspring

    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               eval_budget: int,
               max_generations: int,
               verbose: int = 1) -> GPModelPopulation:
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

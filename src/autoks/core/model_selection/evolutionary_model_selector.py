import warnings
from typing import List, Callable, Optional

import numpy as np

from src.autoks.backend.kernel import compute_kernel
from src.autoks.backend.model import log_likelihood_normalized, RawGPModelType
from src.autoks.core.covariance import Covariance, centered_alignment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.gp_models import gp_regression
from src.autoks.core.grammar import EvolutionaryGrammar
from src.autoks.core.kernel_encoding import tree_to_kernel, KernelNode
from src.autoks.core.model_selection.base import ModelSelector, SurrogateBasedModelSelector
from src.evalg.fitness import shared_fitness_scores
from src.evalg.genprog import HalfAndHalfMutator, SubtreeExchangeLeafBiasedRecombinator
from src.evalg.genprog.generators import BinaryTreeGenerator, HalfAndHalfGenerator
from src.evalg.selection import ExponentialRankingSelector, TruncationSelector, FitnessProportionalSelector
from src.evalg.vary import CrossoverVariator, MutationVariator, CrossMutPopOperator


class EvolutionaryModelSelector(ModelSelector):

    def __init__(self,
                 grammar: Optional[EvolutionaryGrammar] = None,
                 fitness_fn: Callable[[RawGPModelType], float] = log_likelihood_normalized,
                 initializer: Optional[BinaryTreeGenerator] = None,
                 n_init_trees: int = 10,
                 n_parents: int = 10,
                 m_prob: float = 0.10,
                 cx_prob: float = 0.60,
                 pop_size: int = 25,
                 additive_form: bool = False,
                 fitness_sharing: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Callable = gp_regression,
                 gp_args: Optional[dict] = None):
        if grammar is None:
            variation_pct = m_prob + cx_prob  # 60% of individuals created using crossover and 10% mutation
            n_offspring = int(variation_pct * pop_size)
            n_parents = n_offspring
            grammar = self._create_default_grammar(n_offspring, cx_prob, m_prob)
        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer, n_restarts_optimizer)

        if initializer is None:
            initializer = HalfAndHalfGenerator(max_depth=1)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = pop_size
        self.fitness_sharing = fitness_sharing

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

            new_models = self.propose_new_models(population, x, y, verbose=verbose)
            self.expansion_callback(new_models, self, x, y)
            population.update(new_models)

            self.evaluate_models(population.candidates(), x, y, eval_budget, verbose=verbose)

            population.models = self.select_offspring(population)
            self.active_set_callback(population.models, self, x, y)

            depth += 1

        return population

    def select_parents(self, population: ActiveModelPopulation, x, y) -> List[GPModel]:
        """Select parents to expand.

        Here, exponential ranking selection is used.
        """
        selector = ExponentialRankingSelector(self.n_parents, c=0.7)
        raw_fitness_scores = population.fitness_scores()
        models = np.array(population.models)
        if self.fitness_sharing:
            if not isinstance(selector, FitnessProportionalSelector):
                warnings.warn('When using fitness sharing, fitness proportional selection is assumed.')
            individuals = [compute_kernel(gp_model.covariance.raw_kernel, x) for gp_model in models]
            metric = centered_alignment
            effective_fitness_scores = shared_fitness_scores(individuals, raw_fitness_scores, metric)
        else:
            effective_fitness_scores = np.array(raw_fitness_scores)

        return list(selector.select(models, effective_fitness_scores))

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
            operators = self.grammar.operators
            operands = [cov.raw_kernel for cov in self.grammar.base_kernels]
            trees = [self.initializer.generate(operators, operands) for _ in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            covariances = [Covariance(k) for k in kernels]

            return covariances
        else:
            return self.grammar.base_kernels

    @staticmethod
    def _create_default_grammar(n_offspring, cx_prob, m_prob):
        mutator = HalfAndHalfMutator(binary_tree_node_cls=KernelNode, max_depth=1)
        recombinator = SubtreeExchangeLeafBiasedRecombinator()

        cx_variator = CrossoverVariator(recombinator, n_offspring=n_offspring, c_prob=cx_prob)
        mut_variator = MutationVariator(mutator, m_prob=m_prob)
        variators = [cx_variator, mut_variator]
        pop_operator = CrossMutPopOperator(variators)

        return EvolutionaryGrammar(population_operator=pop_operator)


class SurrogateEvolutionaryModelSelector(SurrogateBasedModelSelector):
    grammar: EvolutionaryGrammar

    def __init__(self,
                 grammar: Optional[EvolutionaryGrammar] = None,
                 fitness_fn: Callable[[RawGPModelType], float] = log_likelihood_normalized,
                 query_strategy=None,
                 initializer: Optional[BinaryTreeGenerator] = None,
                 n_init_trees: int = 10,
                 n_parents: int = 10,
                 max_offspring: int = 25,
                 additive_form: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Callable = gp_regression,
                 gp_args: Optional[dict] = None):
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
            operators = self.grammar.operators
            operands = [cov.raw_kernel for cov in self.grammar.base_kernels]
            trees = [self.initializer.generate(operators, operands) for _ in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            covariances = [Covariance(k) for k in kernels]

            return covariances
        else:
            return self.grammar.base_kernels

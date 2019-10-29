import warnings
from time import time
from typing import Callable, Union, List, Optional

import numpy as np

from src.autoks.backend.kernel import compute_kernel
from src.autoks.backend.model import RawGPModelType
from src.autoks.callbacks import CallbackList
from src.autoks.core.covariance import Covariance, centered_alignment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import EvolutionaryGrammar
from src.autoks.core.hyperprior import HyperpriorMap, boms_hyperpriors
from src.autoks.core.kernel_encoding import KernelNode
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.debugging import assert_valid_kernel_kernel
from src.autoks.distance.distance import ActiveModels
from src.autoks.gp_regression_models import KernelKernelGPModel
from src.evalg.fitness import shared_fitness_scores
from src.evalg.genprog import HalfAndHalfMutator, SubtreeExchangeRecombinator
from src.evalg.genprog.generators import BinaryTreeGenerator, HalfAndHalfGenerator
from src.evalg.selection import TruncationSelector, FitnessProportionalSelector
from src.evalg.vary import CrossoverVariator, MutationVariator, CrossMutPopOperator


class EvolutionaryModelSelector(ModelSelector):

    def __init__(self,
                 grammar: Optional[EvolutionaryGrammar] = None,
                 base_kernel_names: Optional[List[str]] = None,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
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
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None,
                 hyperpriors: Optional[HyperpriorMap] = None):
        if grammar is None:
            variation_pct = m_prob + cx_prob  # 60% of individuals created using crossover and 10% mutation
            n_offspring = int(variation_pct * pop_size)
            n_parents = n_offspring
            hyperpriors = hyperpriors or boms_hyperpriors()
            grammar = self._create_default_grammar(n_offspring, cx_prob, m_prob, base_kernel_names=base_kernel_names,
                                                   hyperpriors=hyperpriors)
        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer, n_restarts_optimizer)

        if initializer is None:
            initializer = HalfAndHalfGenerator(max_depth=1)
        self.initializer = initializer
        self.n_init_trees = n_init_trees
        self.max_offspring = pop_size
        self.fitness_sharing = fitness_sharing

    def _train(self,
               eval_budget: int,
               max_generations: int,
               callbacks: CallbackList,
               verbose: int = 1) -> GPModelPopulation:
        population = self._initialize(eval_budget, verbose=verbose, callbacks=callbacks)

        depth = 0
        while self.n_evals < eval_budget:
            callbacks.on_generation_begin(generation=depth, logs={'gp_models': population.models})

            if depth > max_generations:
                break

            self._print_search_summary(depth, population, eval_budget, max_generations, verbose=verbose)

            new_models = self._propose_new_models(population, callbacks=callbacks, verbose=verbose)
            population.update(new_models)

            self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose)
            population.models = self.select_offspring(population)

            callbacks.on_generation_end(generation=depth, logs={'gp_models': population.models})
            depth += 1

        return population

    def _select_parents(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select parents to expand.

        Here, fitness proportional selection is used.
        """
        selector = FitnessProportionalSelector(self.n_parents)
        raw_fitness_scores = population.fitness_scores()
        models = np.array(population.models)
        if self.fitness_sharing:
            if not isinstance(selector, FitnessProportionalSelector):
                warnings.warn('When using fitness sharing, fitness proportional selection is assumed.')
            individuals = [compute_kernel(gp_model.covariance.raw_kernel, self._x_train) for gp_model in models]
            metric = lambda k1, k2: 1 - centered_alignment(k1, k2)
            effective_fitness_scores = shared_fitness_scores(individuals, raw_fitness_scores, metric)
        else:
            effective_fitness_scores = np.array(raw_fitness_scores)

        return list(selector.select(models, effective_fitness_scores))

    def select_offspring(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select offspring for the next generation.

        Here, the top k offspring are chosen to seed the next generation.
        """
        selector = TruncationSelector(self.max_offspring)

        # Make sure best fitness never decreases
        prev_best_fitness = population.best_fitness()

        offspring = list(selector.select(np.array(population.models), np.array(population.fitness_scores())).tolist())

        # Debugging
        scores = [m.score for m in offspring if not m.failed_fitting]
        new_best_fitness = offspring[int(np.nanargmax(scores))].score

        fitness_decreased = (new_best_fitness - prev_best_fitness) < 0
        if fitness_decreased:
            warnings.warn(f"Best fitness decreased from {prev_best_fitness} to {new_best_fitness}")
            assert False

        return offspring

    def _get_initial_candidate_covariances(self) -> List[Covariance]:
        if self.initializer is not None:
            # Generate trees
            # operators = self.grammar.operators
            # operands = [cov.raw_kernel for cov in self.grammar.base_kernels]
            # trees = [self.initializer.generate(operators, operands) for _ in range(self.n_init_trees)]
            # kernels = [tree_to_kernel(tree) for tree in trees]
            # covariances = [Covariance(k) for k in kernels]
            # return covariances
            return self.grammar.base_kernels
        else:
            return self.grammar.base_kernels

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict['initializer'] = self.initializer.to_dict()
        input_dict['n_init_trees'] = self.n_init_trees
        input_dict['max_offspring'] = self.max_offspring
        input_dict['fitness_sharing'] = self.fitness_sharing
        return input_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        standardize_x = input_dict.pop('standardize_x')
        standardize_y = input_dict.pop('standardize_y')

        model_selector = super()._build_from_input_dict(input_dict)

        model_selector.standardize_x = standardize_x
        model_selector.standardize_y = standardize_y

        return model_selector

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        input_dict['pop_size'] = input_dict.pop('max_offspring')
        input_dict['initializer'] = BinaryTreeGenerator.from_dict(input_dict['initializer'])
        return input_dict

    @staticmethod
    def _create_default_grammar(n_offspring: int,
                                cx_prob: float,
                                m_prob: float,
                                base_kernel_names: Optional[List[str]] = None,
                                hyperpriors: Optional[HyperpriorMap] = None):
        mutator = HalfAndHalfMutator(binary_tree_node_cls=KernelNode, max_depth=1)
        recombinator = SubtreeExchangeRecombinator()  # OnePointStrictRecombinator()

        cx_variator = CrossoverVariator(recombinator, n_offspring=n_offspring, c_prob=cx_prob)
        mut_variator = MutationVariator(mutator, m_prob=m_prob)
        variators = [cx_variator, mut_variator]
        pop_operator = CrossMutPopOperator(variators)

        return EvolutionaryGrammar(population_operator=pop_operator, base_kernel_names=base_kernel_names,
                                   hyperpriors=hyperpriors)

    def __str__(self):
        return self.name


class BoemsSurrogateSelector(EvolutionaryModelSelector):

    def __init__(self,
                 active_models: ActiveModels,
                 acquisition_fn: Callable,
                 kernel_kernel_model: KernelKernelGPModel,
                 **ms_args):
        ms_args.update({'hyperpriors': boms_hyperpriors()})
        super().__init__(**ms_args)
        # Don't standardize because input data are indices, output data already standardized.
        self.standardize_x = False
        self.standardize_y = False

        self.active_models = active_models
        self.acquisition_fn = acquisition_fn
        self.kernel_kernel_model = kernel_kernel_model

    def _train(self,
               eval_budget: int,
               max_generations: int,
               callbacks: CallbackList,
               verbose: int = 1) -> GPModelPopulation:
        assert self.active_models.max_n_models >= eval_budget

        population = self._initialize(eval_budget, verbose=verbose, callbacks=callbacks)

        depth = 0
        while self.n_evals < eval_budget:
            callbacks.on_generation_begin(generation=depth, logs={'gp_models': population.models})

            if depth > max_generations:
                break

            self._print_search_summary(depth, population, eval_budget, max_generations, verbose=verbose)

            new_models = self._propose_new_models(population, callbacks=callbacks, verbose=verbose)
            population.update(new_models)

            # update active models
            new_candidates_indices, all_candidate_indices = self.update_population(population)

            # Compute EI for new candidates.
            t0 = time()
            self._evaluate_candidates(new_candidates_indices, verbose=verbose,
                                      eval_budget=eval_budget)
            self.total_eval_time += time() - t0

            # Set remove priority after evaluation.
            self._set_remove_priority(all_candidate_indices)


            population.models = self.select_offspring(population)

            callbacks.on_generation_end(generation=depth, logs={'gp_models': population.models})
            depth += 1

        # Debugging checks.
        if all(s == 0 for s in population.fitness_scores()):
            warnings.warn("All fitness scores are 0")

        # Assert all models in selected_models have EI of 0.
        for m in self.selected_models:
            if m in population.models:
                assert m.score == 0

        return population

    def _initialize(self,
                    eval_budget: int,
                    callbacks: CallbackList,
                    verbose: int = 0) -> ActiveModelPopulation:
        population = ActiveModelPopulation()

        initial_candidates = self._covariances_to_gp_models(self._get_initial_candidate_covariances())
        population.update(initial_candidates)

        new_candidates_indices, all_candidate_indices = self.update_population(population)

        # Compute EI for new candidates.
        t0 = time()
        self._evaluate_candidates(new_candidates_indices, verbose=verbose,
                                  eval_budget=eval_budget)
        self.total_eval_time += time() - t0

        # Set fitness scores of all selected indices to 0
        # Assume exact function evaluation
        for i in self.active_models.selected_indices:
            self.active_models.models[i].score = 0.

        # Set remove priority.
        self._set_remove_priority(all_candidate_indices)

        return population

    def update_population(self, population: GPModelPopulation):
        # update active models
        candidates = population.candidates()
        models_existing = [self.active_models.models[self.active_models.get(c, None)] for c in candidates if
                           self.active_models.get(c, None) is not None]
        new_candidates_indices = self.active_models.update(candidates)

        # Pool of models.
        selected_indices = self.active_models.selected_indices
        all_candidate_indices = set(range(len(self.active_models)))
        all_candidate_indices = list(all_candidate_indices - set(selected_indices))

        # First step is to precompute information for the new candidate models
        kernel_builder = self.kernel_kernel_model.model.kern.distance_builder
        kernel_builder.precompute_information(self.active_models, new_candidates_indices, self._x_train)

        # ii) new candidate models vs all trained models
        kernel_builder.compute_distance(self.active_models, selected_indices, new_candidates_indices)

        # Make sure all necessary indices are not NaN.
        assert_valid_kernel_kernel(kernel_builder, len(self.active_models), self.active_models.selected_indices,
                                   self.active_models.get_candidate_indices())

        # Train the GP.
        self.kernel_kernel_model.model.kern.n_models = len(self.active_models)

        for m in population.models:
            try:
                global_index = self.active_models.index(m)

                if global_index not in all_candidate_indices and np.isnan(m.score):
                    m.score = 0.
                # if it previously was a candidate, assume it keeps previous acq score
                # TODO: test this
                if global_index not in new_candidates_indices and np.isnan(m.score):
                    m.score = self.active_models.models[global_index].score

            except KeyError:
                if np.isnan(m.score):
                    # model was removed from active models, but is still in population
                    score = [m.score for m in models_existing if m.covariance.symbolic_expanded_equals(m.covariance)][0]
                    m.score = score

        return new_candidates_indices, all_candidate_indices

    def _set_remove_priority(self, all_candidate_indices: List[int]):
        acquisition_function_values = [model.score for model in self.active_models.get_candidates()]
        # Convert NaNs to -inf only when evaluation budget is reached.
        np.nan_to_num(acquisition_function_values, nan=np.NINF, copy=False)
        indices_acquisition = np.argsort(np.array(acquisition_function_values).flatten())
        self.active_models.remove_priority = [all_candidate_indices[i] for i in indices_acquisition]

    def _evaluate_candidates(self, candidate_indices: List[int], eval_budget: int, verbose: int = 0) -> List[float]:
        # TODO: rewrite this to work with super class `evaluate_models`.
        x_test = np.array(candidate_indices)[:, None]
        fitness_scores = self.acquisition_fn(x_test, self.kernel_kernel_model).tolist()
        for i, score in zip(candidate_indices, fitness_scores):
            if self.n_evals >= eval_budget:
                if verbose == 3:
                    print('Stopping optimization and evaluation. Evaluation budget reached.\n')
                break

            self.n_evals += 1
            self.visited.add(self.active_models.models[i].covariance.symbolic_expr_expanded)
            self.active_models.models[i].score = score[0]

            if verbose:
                self.pbar.update()

        return fitness_scores

import warnings
from time import time
from typing import Callable, Union, List, Optional

import numpy as np
from GPy.core.parameterization.priors import Gaussian
from GPy.kern import RBFDistanceBuilderKernelKernel

from src.autoks.acquisition import compute_ei
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
from src.autoks.distance.distance import FrobeniusDistanceBuilder, ActiveModels
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


class SurrogateEvolutionaryModelSelector(EvolutionaryModelSelector):

    def __init__(self,
                 selected_models: List[GPModel],
                 fitness_scores: List[float],
                 data_x: np.ndarray,
                 covariance_to_info_map: dict,
                 **ms_args):
        ms_args.update({'hyperpriors': boms_hyperpriors()})
        super().__init__(**ms_args)

        # Copy selected models.
        covariance_dicts = [m.covariance.to_dict() for m in selected_models]
        covariances = [Covariance.from_dict(d) for d in covariance_dicts]
        self.selected_models = self._covariances_to_gp_models(covariances)
        self.fitness_scores = fitness_scores
        self.covariance_to_info_map = covariance_to_info_map

        self.standardize_x = False
        self.standardize_y = False

        # FIXME: these should not be hard-coded.
        max_n_models = 1000
        num_samples = 20
        max_num_hyperparameters = 40
        noise_prior = Gaussian(np.log(0.01), np.sqrt(0.1))

        meta_x_init = np.array(list(range(len(selected_models))))[:, None]
        meta_y_init = np.array(fitness_scores)[:, None]

        # Check shapes of meta_x and meta_y
        assert meta_x_init.ndim == 2 and meta_y_init.ndim == 2
        assert meta_x_init.shape == (len(selected_models), 1) and meta_y_init.shape == (len(fitness_scores), 1)
        assert meta_x_init.shape == meta_y_init.shape

        # Set hyperpriors for selected models.
        if ms_args['hyperpriors']:
            for m in self.selected_models:
                m.covariance.set_hyperpriors(ms_args['hyperpriors'])

        self.active_models = ActiveModels(max_n_models)
        newly_inserted_indices = self.active_models.update(self.selected_models)
        self.active_models.selected_indices = newly_inserted_indices

        self.kernel_to_index_map = dict()  # kernel index in kernel builder matrix
        for i in newly_inserted_indices:
            self.kernel_to_index_map[GPModelPopulation._hash_model(self.active_models.models[i])] = i

        initial_candidate_indices = list(meta_x_init.flatten().tolist())
        self.kernel_builder = FrobeniusDistanceBuilder(noise_prior,
                                                       num_samples,
                                                       max_num_hyperparameters,
                                                       max_n_models,
                                                       self.active_models,
                                                       initial_candidate_indices,
                                                       data_x)
        # Compute K_XX.
        selected_ind = self.active_models.selected_indices
        self.kernel_builder.compute_distance(self.active_models, selected_ind, selected_ind)

        #### set info map
        for model in self.active_models.get_selected_models():
            self.covariance_to_info_map[GPModelPopulation._hash_model(model)] = model.info
        ####

        assert_valid_kernel_kernel(self.kernel_builder, len(self.active_models), self.active_models.selected_indices,
                                   self.active_models.get_candidate_indices())

        # TODO: figure out better initial lengthscale guess
        ell = np.nanmax(self.kernel_builder.get_kernel(len(self.active_models))) * 10
        kernel_kernel = Covariance(RBFDistanceBuilderKernelKernel(self.kernel_builder, n_models=len(self.active_models),
                                                                  lengthscale=ell))
        kernel_kernel_hyperpriors = ms_args['hyperpriors']
        self.kernel_kernel_gp_model = KernelKernelGPModel(meta_x_init, meta_y_init, kernel_kernel, verbose=False,
                                                          exact_f_eval=False,
                                                          kernel_kernel_hyperpriors=kernel_kernel_hyperpriors)

        self.kernel_kernel_gp_model_train_freq = 10

    def _initialize(self,
                    eval_budget: int,
                    callbacks: CallbackList,
                    verbose: int = 0) -> ActiveModelPopulation:
        population = ActiveModelPopulation()

        initial_candidates = self._covariances_to_gp_models(self._get_initial_candidate_covariances())
        all_candidate_indices, new_candidates_indices = self._update_gp_model_pop(population, initial_candidates, 0)

        self._set_acq_scores_to_zero(population, all_candidate_indices)

        # Compute EI for new candidates.
        t0 = time()
        self._evaluate_candidates(new_candidates_indices, verbose=verbose,
                                  eval_budget=eval_budget)
        self.total_eval_time += time() - t0

        # Set remove priority.
        acquisition_function_values = [model.score for model in self.active_models.get_candidates()]
        indices_acquisition = np.argsort(np.array(acquisition_function_values).flatten())
        self.active_models.remove_priority = [all_candidate_indices[i] for i in indices_acquisition]

        return population

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

            all_candidate_indices, new_candidates_indices = self._update_gp_model_pop(population, new_models, depth + 1)

            self._set_acq_scores_to_zero(population, all_candidate_indices)

            # Compute EI for new candidates.

            t0 = time()
            self._evaluate_candidates(new_candidates_indices, verbose=verbose,
                                      eval_budget=eval_budget)
            self.total_eval_time += time() - t0

            # Set remove priority.
            acquisition_function_values = [model.score for model in self.active_models.get_candidates()]
            indices_acquisition = np.argsort(np.array(acquisition_function_values).flatten())
            self.active_models.remove_priority = [all_candidate_indices[i] for i in indices_acquisition]

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

    def _evaluate_candidates(self, candidate_indices: List[int], eval_budget: int, verbose: int = 0) -> List[float]:
        # TODO: rewrite this to work with super class `evaluate_models`.
        x_test = np.array(candidate_indices)[:, None]
        fitness_scores = compute_ei(x_test, self.kernel_kernel_gp_model).tolist()
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

    def _update_gp_model_pop(self,
                             population: GPModelPopulation,
                             candidates: List[GPModel],
                             generation: int):
        """Update GP model population and kernel kernel GP model."""
        population.update(candidates)

        # Update active models.
        candidate_models = population.candidates()
        candidate_models = [m for m in candidate_models if
                            GPModelPopulation._hash_model(m) not in self.kernel_to_index_map]
        new_candidates_indices = self.active_models.update(candidate_models)

        # Pool of models.
        selected_indices = self.active_models.selected_indices
        all_candidate_indices = set(range(len(self.active_models)))
        all_candidate_indices = list(all_candidate_indices - set(selected_indices))

        # Update kernel index dict.
        for i in all_candidate_indices:
            self.kernel_to_index_map[GPModelPopulation._hash_model(self.active_models.models[i])] = i

        # Update model distances using the kernel builder.
        # self.kernel_builder.update(self.active_models, new_candidates_indices, all_candidate_indices, selected_indices,
        #                            self._x_train)
        active_models, all_candidates_indices, data_X = self.active_models, all_candidate_indices, self._x_train
        ##############################
        # First step is to precompute information for the new candidate models
        new_candidates_indice_no_info = []
        for i in new_candidates_indices:
            model = self.active_models.models[i]
            key = GPModelPopulation._hash_model(model)
            if key not in self.covariance_to_info_map:
                new_candidates_indice_no_info.append(i)

        # set info for new candidates
        new_candidates_indices_with_global_info = list(set(new_candidates_indices) - set(new_candidates_indice_no_info))
        for i in new_candidates_indices_with_global_info:
            model = self.active_models.models[i]
            key = GPModelPopulation._hash_model(model)
            info = self.covariance_to_info_map[key]
            model.info = info

        self.kernel_builder.precompute_information(active_models, new_candidates_indice_no_info, data_X)

        # save  newly created info!
        for i in new_candidates_indice_no_info:
            model = self.active_models.models[i]
            self.covariance_to_info_map[GPModelPopulation._hash_model(model)] = model.info
        # ii) new candidate models vs all trained models
        self.kernel_builder.compute_distance(active_models, selected_indices, new_candidates_indices)
        ##############################

        # Make sure all necessary indices are not NaN.
        assert_valid_kernel_kernel(self.kernel_builder, len(self.active_models), self.active_models.selected_indices,
                                   self.active_models.get_candidate_indices())

        # Train the GP.
        self.kernel_kernel_gp_model.model.kern.n_models = len(self.active_models)
        if generation % self.kernel_kernel_gp_model_train_freq == 0:
            self.kernel_kernel_gp_model.train()

        return all_candidate_indices, new_candidates_indices

    def _set_acq_scores_to_zero(self,
                                population: GPModelPopulation,
                                all_candidate_indices: List[int]):
        """Set acquisition score of evaluated models to 0 (exact objective evaluations assumed)"""
        for m in population.models:
            # FIXME: don't call protected method.
            key = GPModelPopulation._hash_model(m)
            global_index = self.kernel_to_index_map[key]
            if global_index not in all_candidate_indices and np.isnan(m.score):
                m.score = 0.

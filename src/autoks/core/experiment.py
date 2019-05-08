import warnings
from time import time
from typing import Callable, List, Tuple, Optional, FrozenSet, Union, Any, Type

import matplotlib.pyplot as plt
import numpy as np
from GPy.core import GP
from GPy.core.parameterization.priors import Gaussian
from GPy.inference.latent_function_inference import Laplace
from GPy.kern import RBFKernelKernel
from GPy.models import GPRegression
from numpy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler

from src.autoks.backend.kernel import set_priors, sort_kernel, get_all_1d_kernels, n_base_kernels, \
    kernel_to_infix, KERNEL_DICT
from src.autoks.backend.model import set_model_kern, is_nan_model, log_likelihood_normalized, AIC, BIC, pl2
from src.autoks.core.acquisition_function import ExpectedImprovementPerSec
from src.autoks.core.active_set import ActiveSet
from src.autoks.core.covariance import all_pairs_avg_dist, pairwise_centered_alignments
from src.autoks.core.gp_model import remove_duplicate_gp_models, GPModel, pretty_print_gp_models
from src.autoks.core.grammar import BaseGrammar, BOMSGrammar, CKSGrammar, EvolutionaryGrammar, RandomGrammar
from src.autoks.core.hyperprior import Hyperpriors, boms_hyperpriors
from src.autoks.core.kernel_encoding import KernelNode
from src.autoks.core.kernel_selection import KernelSelector, BOMS_kernel_selector, CKS_kernel_selector, \
    evolutionary_kernel_selector
from src.autoks.core.query_strategy import NaiveQueryStrategy, QueryStrategy, BOMSInitQueryStrategy, BestScoreStrategy
from src.autoks.distance.distance import HellingerDistanceBuilder, DistanceBuilder
from src.autoks.gp_regression_models import KernelKernelGPRegression
from src.autoks.plotting import plot_kernel_diversity_summary, plot_best_scores, plot_score_summary, \
    plot_n_hyperparams_summary, plot_n_operands_summary, plot_base_kernel_freqs, plot_cov_dist_summary, plot_kernel_tree
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_svr, rmse_lin_reg, rmse_rbf, rmse_knn, \
    ExperimentReportGenerator
from src.autoks.statistics import StatBookCollection, Statistic, StatBook
from src.autoks.util import type_count, pretty_time_delta
from src.evalg.genprog import HalfAndHalfMutator, HalfAndHalfGenerator, \
    SubtreeExchangeLeafBiasedRecombinator
from src.evalg.vary import CrossoverVariator, MutationVariator, CrossMutPopOperator


class Experiment:
    distance_builder: Optional[DistanceBuilder]
    grammar: BaseGrammar
    kernel_selector: KernelSelector
    objective: Callable[[GP], float]
    kernel_families: List[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    standardize_x: bool
    standardize_y: bool
    eval_budget: int
    max_depth: Optional[int]
    hyperpriors: Optional[Hyperpriors]
    init_query_strat: Optional[QueryStrategy]
    query_strat: Optional[QueryStrategy]
    gp_model: Optional[GP]
    additive_form: bool
    debug: bool
    verbose: bool
    tabu_search: bool
    optimizer: Optional[str]
    n_restarts_optimizer: int
    max_null_queries: int
    max_same_expansions: int
    use_surrogate: bool
    kernel_kernel: RBFKernelKernel
    surrogate_model: Optional
    surrogate_model_cls: Type
    surrogate_opt_freq: int

    def __init__(self, grammar, kernel_selector, objective, x_train, y_train, x_test, y_test,
                 standardize_x=True, standardize_y=True, eval_budget=50, max_depth=None, gp_model=None,
                 init_query_strat=None, query_strat=None, additive_form=False, debug=False,
                 verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3, optimizer=None,
                 n_restarts_optimizer=10, use_surrogate=True, use_laplace=True):
        self.grammar = grammar
        self.kernel_selector = kernel_selector
        self.objective = objective

        self.x_train = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
        # self.x_train = np.atleast_2d(x_train)
        if x_test is not None:
            self.x_test = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
        else:
            self.x_test = None
        # self.x_test = np.atleast_2d(x_test)
        # Make y >= 2-dimensional
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        # self.y_train = np.atleast_2d(y_train)
        if y_test is not None:
            self.y_test = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
        else:
            self.y_test = None
        # self.y_test = np.atleast_2d(y_test)
        # only save scaled version of data
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        if standardize_x:
            scaler = StandardScaler()
            if standardize_x:
                self.x_train = scaler.fit_transform(self.x_train)
                if self.x_test is not None:
                    self.x_test = scaler.transform(self.x_test)
            # this is handled in Model!
            # if standardize_y:
            #     self.y_train = scaler.fit_transform(self.y_train)
            #     if self.y_test is not None:
            #         self.y_test = scaler.transform(self.y_test)
        self.n_dims = self.x_train.shape[1]

        self.eval_budget = eval_budget  # number of model evaluations (budget)

        if max_depth is None:
            # By default, the model search is terminated only when the evaluation budget is expended.
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth

        self.n_evals = 0
        self.additive_form = additive_form
        self.debug = debug
        self.verbose = verbose
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer

        # statistics used for plotting
        self.n_hyperparams_name = 'n_hyperparameters'
        self.n_operands_name = 'n_operands'
        self.base_kern_freq_names = [base_kern_name + '_frequency' for base_kern_name in self.grammar.base_kernel_names]
        self.score_name = 'score'
        self.cov_dists_name = 'cov_dists'
        self.diversity_scores_name = 'diversity_scores'
        self.best_stat_name = 'best'
        # All stat books track these variables
        shared_multi_stat_names = [self.n_hyperparams_name, self.n_operands_name] + self.base_kern_freq_names

        # raw value statistics
        base_kern_stat_funcs = [base_kern_freq(base_kern_name) for base_kern_name in self.grammar.base_kernel_names]
        shared_stats = [get_n_hyperparams, get_n_operands] + base_kern_stat_funcs

        self.evaluations_name = 'evaluations'
        self.active_set_name = 'active_set'
        self.expansion_name = 'expansion'
        stat_book_names = [self.evaluations_name, self.expansion_name, self.active_set_name]
        self.stat_book_collection = StatBookCollection(stat_book_names, shared_multi_stat_names, shared_stats)

        sb_active_set = self.stat_book_collection.stat_books[self.active_set_name]
        sb_active_set.add_raw_value_stat(self.score_name, get_model_scores)
        sb_active_set.add_raw_value_stat(self.cov_dists_name, get_cov_dists)
        sb_active_set.add_raw_value_stat(self.diversity_scores_name, get_diversity_scores)
        sb_active_set.multi_stats[self.n_hyperparams_name].add_statistic(Statistic(self.best_stat_name,
                                                                                   get_best_n_hyperparams))
        sb_active_set.multi_stats[self.n_operands_name].add_statistic(Statistic(self.best_stat_name,
                                                                                get_best_n_operands))

        sb_evals = self.stat_book_collection.stat_books[self.evaluations_name]
        sb_evals.add_raw_value_stat(self.score_name, get_model_scores)
        # sb_evals.multi_stats[n_hyperparams_name].add_statistic(Statistic(best_stat_name, get_best_n_hyperparams))
        # sb_evals.multi_stats[n_operands_name].add_statistic(Statistic(best_stat_name, get_best_n_operands))

        self.total_optimization_time = 0
        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_model_search_time = 0
        self.total_query_time = 0

        # self.hyperpriors = hyperpriors

        if gp_model is not None:
            # do not use hyperpriors if gp model is given.
            self.gp_model = gp_model
        else:
            # default model is GP Regression
            self.gp_model = GPRegression(self.x_train, self.y_train, normalizer=standardize_y)

            if use_laplace:
                self.gp_model.inference_method = Laplace()

            if self.grammar.hyperpriors is not None:
                # set likelihood hyperpriors
                likelihood_priors = self.grammar.hyperpriors['GP']
                self.gp_model.likelihood = set_priors(self.gp_model.likelihood, likelihood_priors)
            # randomize likelihood
            self.gp_model.likelihood.randomize()

        if init_query_strat is not None:
            self.init_query_strat = init_query_strat
        else:
            self.init_query_strat = NaiveQueryStrategy()

        if query_strat is not None:
            self.query_strat = query_strat
        else:
            self.query_strat = NaiveQueryStrategy()

        self.tabu_search = tabu_search

        # Kernel search termination conditions.
        self.max_same_expansions = max_same_expansions  # Maximum number of same kernel proposal before terminating
        self.max_null_queries = max_null_queries  # Maximum number of empty queries in a row allowed before terminating

        self.use_surrogate = use_surrogate
        self.surrogate_model = None
        # TODO: ensure this matches distance builder limit
        self.active_set = ActiveSet(max_n_models=1000)

        # Used for expected improvement per second.
        self.n_kernel_params = []
        self.objective_times = []

        # visited set of all expanded kernel expressions previously evaluated
        self.visited = set()

    def model_search(self) -> List[GPModel]:
        """Perform automated kernel search.

        :return: list of models
        """
        t_init = time()

        # initialize models
        kernels = self.grammar.initialize()
        kernels = remove_duplicate_gp_models(kernels)  # TODO: clean up
        kernels = self.randomize_models(kernels)

        # convert to additive form if necessary
        if self.additive_form:
            for gp_model in kernels:
                gp_model.covariance.to_additive_form()

        # create distance builder if using surrogate model
        if self.use_surrogate:
            new_candidate_indices = self.active_set.update(kernels)
            assert new_candidate_indices == self.active_set.get_candidate_indices()
            # for now, it must be a Hellinger distance builder
            self.distance_builder = self.create_hellinger_db(self.active_set, self.x_train)

        # Select gp_models by acquisition function to be evaluated
        selected_kernels, ind, acq_scores = self.query_models(kernels, self.init_query_strat, self.grammar.hyperpriors)
        unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
        unselected_kernels = [unevaluated_kernels[i] for i in range(len(unevaluated_kernels)) if i not in ind]
        budget_left = self.eval_budget - self.n_evals
        newly_evaluated_kernels = self.opt_and_eval_models(selected_kernels[:budget_left])
        old_evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated and kernel not in selected_kernels]
        kernels = newly_evaluated_kernels + unselected_kernels + old_evaluated_kernels
        for gp_model in newly_evaluated_kernels:
            self.visited.add(gp_model.covariance.symbolic_expr_expanded)

        if self.use_surrogate:
            new_candidate_indices = []
            new_selected_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel in
                                    newly_evaluated_kernels]
            old_selected_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel in
                                    old_evaluated_kernels]
            all_candidates_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel is not None
                                      and not kernel.evaluated]
            fitness_scores = [self.active_set.models[i].score for i in old_selected_indices] + \
                             [self.active_set.models[i].score for i in new_selected_indices]
            assert self.active_set.selected_indices == old_selected_indices + new_selected_indices
            builder = self.distance_builder
            builder.update_multiple(self.active_set, new_candidate_indices, all_candidates_indices,
                                    old_selected_indices,
                                    new_selected_indices, data_X=self.x_train)
            self.surrogate_model = KernelKernelGPRegression.from_distance_builder(builder, self.active_set,
                                                                                  fitness_scores)
            assert self.active_set.selected_indices == self.surrogate_model.X.flatten().astype(int).tolist()
            assert fitness_scores == self.surrogate_model.Y.flatten().tolist()
            self.optimize_surrogate_model()

        self.update_stat_book(self.stat_book_collection.stat_books[self.active_set_name], kernels)

        prev_expansions = []
        prev_n_queried = []
        depth = 0
        while self.n_evals < self.eval_budget:
            if depth > self.max_depth:
                break

            if self.verbose:
                self.print_search_summary(depth, kernels)

            parents = self.select_parents(kernels)

            # Fix each parent before expansion for use in and initialization optimization, skipping nan-evaluated
            # gp_models
            for parent in self.remove_nan_scored_models(parents):
                parent.covariance.raw_kernel.fix()

            new_kernels = self.propose_new_kernels(parents)
            self.update_stat_book(self.stat_book_collection.stat_books[self.expansion_name], new_kernels)
            # Check for same expansions
            if self.all_same_expansion(new_kernels, prev_expansions, self.max_same_expansions):
                if self.verbose:
                    print(f'Terminating kernel search. The last {self.max_same_expansions} expansions proposed the same'
                          f' gp_models.')
                break
            else:
                prev_expansions = self.update_kernel_infix_set(new_kernels, prev_expansions, self.max_same_expansions)

            new_kernels = self.randomize_models(new_kernels)
            kernels += new_kernels

            # evaluate, prune, and optimize gp_models
            if not self.use_surrogate:
                n_before = len(kernels)
                kernels = remove_duplicate_gp_models(kernels)
                if self.verbose:
                    n_removed = n_before - len(kernels)
                    print(f'Removed {n_removed} duplicate gp_models.\n')

            # update distance builder:
            if self.use_surrogate:
                # update distance builder
                new_candidate_indices = self.active_set.update(new_kernels)
                new_selected_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel in
                                        newly_evaluated_kernels]
                old_selected_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel in
                                        old_evaluated_kernels]
                all_candidates_indices = [i for (i, kernel) in enumerate(self.active_set.models) if kernel is not None
                                          and not kernel.evaluated]
                fitness_scores = [self.active_set.models[i].score for i in old_selected_indices] + \
                                 [self.active_set.models[i].score for i in new_selected_indices]
                assert self.active_set.selected_indices == old_selected_indices + new_selected_indices

                self.surrogate_model.update(self.active_set, new_candidate_indices, all_candidates_indices,
                                            old_selected_indices, new_selected_indices, self.x_train, fitness_scores)
                assert self.active_set.selected_indices == self.surrogate_model.X.flatten().astype(int).tolist()
                assert fitness_scores == self.surrogate_model.Y.flatten().tolist()
                self.optimize_surrogate_model()
            # Select gp_models by acquisition function to be evaluated
            selected_kernels, ind, acq_scores = self.query_models(kernels, self.query_strat, self.grammar.hyperpriors)

            # Check for empty queries
            prev_n_queried.append(len(ind))
            if all([n == 0 for n in prev_n_queried[-self.max_null_queries:]]) and \
                    len(prev_n_queried) >= self.max_null_queries:
                if self.verbose:
                    print(f'Terminating kernel search. The last {self.max_null_queries} queries were empty.')
                break

            unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
            unselected_kernels = [unevaluated_kernels[i] for i in range(len(unevaluated_kernels)) if i not in ind]
            newly_evaluated_kernels = self.opt_and_eval_models(selected_kernels)
            old_evaluated_kernels = [kernel for kernel in kernels if
                                     kernel.evaluated and kernel not in selected_kernels]
            kernels = newly_evaluated_kernels + unselected_kernels + old_evaluated_kernels
            for gp_model in newly_evaluated_kernels:
                self.visited.add(gp_model.covariance.symbolic_expr_expanded)

            kernels = self.select_offspring(kernels)
            self.update_stat_book(self.stat_book_collection.stat_books[self.active_set_name], kernels)
            depth += 1

        self.total_model_search_time += time() - t_init

        return kernels

    def print_search_summary(self, depth, kernels):
        print(f'Iteration {depth}/{self.max_depth}')
        print(f'Evaluated {self.n_evals}/{self.eval_budget}')
        evaluated_gp_models = [gp_model for gp_model in self.remove_nan_scored_models(kernels)
                               if gp_model.evaluated]
        scores = [gp_model.score for gp_model in evaluated_gp_models]
        arg_max_score = int(np.argmax(scores))
        best_kernel = evaluated_gp_models[arg_max_score]
        sizes = [len(gp_model.covariance.to_binary_tree()) for gp_model in evaluated_gp_models]
        print(f'Avg. objective = %0.6f' % np.mean(scores))
        print(f'Best objective = %.6f' % scores[arg_max_score])
        print(f'Avg. size = %.2f' % np.mean(sizes))
        print('Best kernel:')
        best_kernel.covariance.pretty_print()
        print('')

    def optimize_surrogate_model(self):
        if self.verbose:
            print('Optimizing surrogate model\n')
        self.surrogate_model.optimize()

    def query_models(self,
                     kernels: List[GPModel],
                     query_strategy: QueryStrategy,
                     hyperpriors: Optional[Hyperpriors] = None) \
            -> Tuple[List[GPModel], List[int], List[float]]:
        """Select gp_models using the acquisition function of the query strategy.

        :param kernels:
        :param query_strategy:
        :param hyperpriors:
        :return:
        """
        t0 = time()
        unevaluated_kernels_ind = [i for (i, kernel) in enumerate(kernels) if not kernel.evaluated]
        unevaluated_kernels = [kernels[i] for i in unevaluated_kernels_ind]
        ind, acq_scores = query_strategy.query(unevaluated_kernels_ind, kernels, self.x_train, self.y_train,
                                               hyperpriors, self.surrogate_model, durations=self.objective_times,
                                               n_hyperparams=self.n_kernel_params)
        selected_kernels = query_strategy.select(np.array(unevaluated_kernels), np.array(acq_scores))
        self.total_query_time += time() - t0

        if self.use_surrogate:
            selected_ind = [i for (i, m) in enumerate(self.active_set.models) if m in selected_kernels]
            self.active_set.selected_indices += list(selected_ind)
            # assert list(selected_kernels) in self.active_set.models
            # TODO: set remove priority only for all candidates
            self.active_set.remove_priority = [unevaluated_kernels_ind[i] for i in np.argsort(acq_scores)]

        if self.verbose:
            n_selected = len(ind)
            plural_suffix = '' if n_selected == 1 else 's'
            print(f'Query strategy selected {n_selected} kernel{plural_suffix}:')

            acq_scores_selected = [s for i, s in enumerate(acq_scores) if i in ind]
            for kern, score in zip(selected_kernels, acq_scores_selected):
                kern.covariance.pretty_print()
                print('\tacq. score =', score)
                # print(str(kern), 'acq. score =', score)
            print('')

        return selected_kernels, ind, acq_scores

    def select_parents(self, kernels: List[GPModel]) -> List[GPModel]:
        """Choose parents to later expand.

        :param kernels:
        :return:
        """
        evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
        if self.tabu_search:
            # Expanded gp_models are the tabu list.
            evaluated_kernels = [kernel for kernel in evaluated_kernels if not kernel.expanded]
        kernel_scores = [kernel.score for kernel in evaluated_kernels]
        parents = self.kernel_selector.select_parents(evaluated_kernels, kernel_scores)
        # Print parent (seed) gp_models
        if self.debug:
            pretty_print_gp_models(parents, 'Parent')

        return parents

    def propose_new_kernels(self, parents: List[GPModel]) -> List[GPModel]:
        """Propose new gp_models using the grammar given a list of parent gp_models.

        :param parents:
        :return:
        """
        # set gp_models to expanded
        for parent in parents:
            parent.expanded = True

        t0_exp = time()
        new_kernels = self.grammar.get_candidates(parents, verbose=self.verbose)
        self.total_expansion_time += time() - t0_exp

        if self.additive_form:
            for gp_model in new_kernels:
                gp_model.covariance.to_additive_form()

        return new_kernels

    def select_offspring(self, kernels: List[GPModel]) -> List[GPModel]:
        """Select next round of gp_models.

        :param kernels:
        :return:
        """
        # scores = [kernel.score for kernel in gp_models]

        # Prioritize keeping evaluated models.
        # TODO: Check correctness
        augmented_scores = [k.score if k.evaluated and not k.nan_scored else -np.inf for k in kernels]

        offspring = self.kernel_selector.select_offspring(kernels, augmented_scores)

        if self.verbose:
            print(f'Offspring selector kept {len(offspring)}/{len(kernels)} gp_models\n')

        return offspring

    def opt_and_eval_models(self, models: List[GPModel]) -> List[GPModel]:
        """Optimize and evaluate all gp_models

        :param models:
        :return:
        """
        evaluated_models = []

        for gp_model in models:
            if self.n_evals >= self.eval_budget:
                if self.verbose:
                    print('Stopping optimization and evaluation. Evaluation budget reached.\n')
                break
            elif gp_model.covariance.symbolic_expr_expanded in self.visited:
                if self.verbose:
                    print('Skipping model because it was previously evaluated')
                    gp_model.covariance.pretty_print()
                    print()
                continue
            t0 = time()

            optimized_model = self.optimize_model(gp_model)

            t1 = time()
            self.total_optimization_time += t1 - t0

            evaluated_model = self.evaluate_model(optimized_model)

            self.total_eval_time += time() - t1
            evaluated_models.append(evaluated_model)

            if not evaluated_model.nan_scored:
                self.update_stat_book(self.stat_book_collection.stat_books[self.evaluations_name], [evaluated_model])

        evaluated_models = self.remove_nan_scored_models(evaluated_models)

        if self.verbose:
            print('Printing all results')
            # Sort models by scores with un-evaluated models last
            for gp_model in sorted(evaluated_models, key=lambda x: (x.score is not None, x.score), reverse=True):
                gp_model.covariance.pretty_print()
                print('\tobjective =', gp_model.score)
            print('')
        return evaluated_models

    def optimize_model(self, gp_model: GPModel) -> GPModel:
        """Optimize the hyperparameters of the model

        All of the parameters which were part of the previous parent model are initialized to their previous
        values. All parameterized are then optimized, randomly restarting the newly introduced parameters.

        :param gp_model:
        :return:
        """
        if not gp_model.evaluated:
            try:
                kernel = gp_model.covariance.raw_kernel
                t0 = time()
                k_unfixed = kernel.copy()
                k_unfixed.unfix()

                # Optimize unfixed kernel (all params unfixed)
                set_model_kern(self.gp_model, k_unfixed)
                self.gp_model.optimize_restarts(ipython_notebook=False, optimizer=self.optimizer,
                                                num_restarts=1, verbose=False, robust=True)

                # Optimize (with restarts) the newly added parameters
                k_fixed = kernel

                # Set param values of fixed kernel to the previously optimized ones of unfixed kernel
                new_params = k_unfixed.param_array
                k_fixed[:] = new_params

                set_model_kern(self.gp_model, k_fixed)
                self.gp_model.optimize_restarts(ipython_notebook=False, optimizer=self.optimizer,
                                                num_restarts=self.n_restarts_optimizer, verbose=False, robust=True)

                # Unfix all params and set kernel
                k_fixed.unfix()
                set_model_kern(self.gp_model, k_fixed)
                gp_model.covariance.raw_kernel = self.gp_model.kern
                gp_model.lik_params = self.gp_model.likelihood[:].copy()
                delta_t = time() - t0
                self.update_object_time_predictor(kernel.size, delta_t)
            except LinAlgError:
                warnings.warn('Y covariance of kernel %s is not positive definite' % gp_model)
        else:
            raise RuntimeError('already optimized')
        return gp_model

    def evaluate_model(self, gp_model: GPModel) -> GPModel:
        """Evaluate a given model using the objective function

        :param gp_model:
        :return:
        """
        if not gp_model.evaluated:
            set_model_kern(self.gp_model, gp_model.covariance.raw_kernel)
            self.gp_model.likelihood[:] = gp_model.lik_params

            # Check if parameters are well-defined:
            gp_model.nan_scored = is_nan_model(self.gp_model)
            if not gp_model.nan_scored:
                score = self.objective(self.gp_model)
                self.n_evals += 1
                # gp_model.evaluated = True
                gp_model.score = score
            else:
                gp_model.score = np.nan
                # also count a nan-evaluated kernel as an evaluation
                # self.n_evals += 1

        return gp_model

    def all_same_expansion(self,
                           new_kernels: List[GPModel],
                           prev_expansions: List[FrozenSet[str]],
                           max_expansions: int) -> bool:
        kernels_infix_new = self.model_to_infix_set(new_kernels)
        all_same = all([s == kernels_infix_new for s in prev_expansions])
        return all_same and len(prev_expansions) == max_expansions

    def model_to_infix_set(self, gp_models: List[GPModel]) -> FrozenSet[str]:
        kernels_sorted = [sort_kernel(gp_model.covariance.raw_kernel) for gp_model in gp_models]
        return frozenset([kernel_to_infix(kernel) for kernel in kernels_sorted])

    def update_kernel_infix_set(self, new_kernels: List[GPModel],
                                prev_expansions: List[FrozenSet[str]],
                                max_expansions: int) -> List[FrozenSet[str]]:
        expansions = prev_expansions.copy()
        if len(prev_expansions) == max_expansions:
            expansions = expansions[1:]
        elif len(prev_expansions) < max_expansions:
            expansions += [self.model_to_infix_set(new_kernels)]

        return expansions

    @staticmethod
    def randomize_models(gp_models: List[GPModel]) -> List[GPModel]:
        for gp_model in gp_models:
            gp_model.covariance.raw_kernel.randomize()

        return gp_models

    def remove_nan_scored_models(self, gp_models: List[GPModel]) -> List[GPModel]:
        """Remove all models that have NaN scores.

        :param gp_models:
        :return:
        """
        return [gp_model for gp_model in gp_models if not gp_model.nan_scored]

    def summarize(self, gp_model: List[GPModel]) -> None:
        """Summarize the experiment.

        :param gp_model:
        :return:
        """
        evaluated_gp_models = [model for model in gp_model if model.evaluated]
        sorted_gp_models = sorted(evaluated_gp_models, key=lambda x: x.score, reverse=True)
        best_gp_model = sorted_gp_models[0]
        best_kernel = best_gp_model.covariance.raw_kernel
        best_model = self.gp_model.__class__(self.x_train, self.y_train, kernel=best_kernel,
                                             normalizer=self.standardize_y)
        # Set the likelihood parameters.
        best_model.likelihood[:] = best_gp_model.lik_params

        # If training data is 1D, show a plot.
        if best_model.input_dim == 1:
            best_model.plot(plot_density=True, title='Best Model')
            plt.show()

        # View results of experiment
        for stat_book in self.stat_book_collection.stat_book_list():
            self.plot_stat_book(stat_book)

        # Plot the kernel tree of the best model
        plot_kernel_tree(best_gp_model)

        print('')
        self.timing_report()
        print('')

        print('Best model:')
        best_gp_model.covariance.pretty_print()

        print('In full form:')
        best_gp_model.covariance.print_full()
        print('')

        # Summarize model
        nll = -best_model.log_likelihood()
        nll_norm = log_likelihood_normalized(best_model)
        if self.x_test is not None:
            mean_nlpd = np.mean(-best_model.log_predictive_density(self.x_test, self.y_test))
        else:
            mean_nlpd = np.nan
        aic = AIC(best_model)
        bic = BIC(best_model)
        pl2_score = pl2(best_model)

        print('NLL = %.3f' % nll)
        print('NLL (normalized) = %.3f' % nll_norm)
        print('NLPD = %.3f' % mean_nlpd)
        print('AIC = %.3f' % aic)
        print('BIC = %.3f' % bic)
        print('PL2 = %.3f' % pl2_score)
        print('')

        # Compare RMSE of best model to other models
        if self.x_test is not None and self.y_test is not None:
            best_model_rmse = compute_gpy_model_rmse(best_model, self.x_test, self.y_test)
            svm_rmse = rmse_svr(self.x_train, self.y_train, self.x_test, self.y_test)
            lr_rmse = rmse_lin_reg(self.x_train, self.y_train, self.x_test, self.y_test)
            se_rmse = rmse_rbf(self.x_train, self.y_train, self.x_test, self.y_test)
            knn_rmse = rmse_knn(self.x_train, self.y_train, self.x_test, self.y_test)

            print('RMSE Best Model = %.3f' % best_model_rmse)
            print('RMSE Linear Regression = %.3f' % lr_rmse)
            print('RMSE SVM = %.3f' % svm_rmse)
            print('RMSE RBF = %.3f' % se_rmse)
            print('RMSE k-NN = %.3f' % knn_rmse)

    def run(self,
            summarize: bool = True,
            create_report: bool = True,
            **kwargs) -> List[GPModel]:
        """Run the model search and optionally summarize and create a report

        :param summarize:
        :param create_report:
        :param kwargs:
        :return:
        """
        gp_model = self.model_search()
        if summarize:
            self.summarize(gp_model)

        if create_report:
            if self.verbose:
                print('')
                print('Creating report...')
            report_gen = ExperimentReportGenerator(self, gp_model, self.x_test, self.y_test)
            report_gen.summarize_experiment(**kwargs)

        return gp_model

    def plot_stat_book(self, stat_book: StatBook):
        ms = stat_book.multi_stats
        x_label = 'evaluations' if stat_book.name == self.evaluations_name else 'generation'
        if self.score_name in ms:
            plot_best_scores(self.score_name, self.evaluations_name, stat_book)
            plot_score_summary(self.score_name, self.evaluations_name, stat_book)
        if self.n_hyperparams_name in ms:
            plot_n_hyperparams_summary(self.n_hyperparams_name, self.best_stat_name, stat_book, x_label)
        if self.n_operands_name in ms:
            plot_n_operands_summary(self.n_operands_name, self.best_stat_name, stat_book, x_label)
        if all(key in ms for key in self.base_kern_freq_names):
            plot_base_kernel_freqs(self.base_kern_freq_names, stat_book, x_label)
        if self.cov_dists_name in ms:
            plot_cov_dist_summary(self.cov_dists_name, stat_book, x_label)
        if self.diversity_scores_name in ms:
            plot_kernel_diversity_summary(self.diversity_scores_name, stat_book, x_label)

    def get_timing_report(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get the runtime report of the kernel search.

        :return:
        """
        eval_time = self.total_eval_time
        opt_time = self.total_optimization_time
        expansion_time = self.total_expansion_time
        query_time = self.total_query_time
        total_time = self.total_model_search_time
        other_time = total_time - eval_time - opt_time - expansion_time - query_time

        labels = ['Evaluation', 'Optimization', 'Expansion', 'Query', 'Other']
        x = np.array([eval_time, opt_time, expansion_time, query_time, other_time])
        x_pct = 100 * (x / total_time)

        return labels, x, x_pct

    def timing_report(self) -> None:
        """Print a runtime report of the model search.

        :return:
        """
        labels, x, x_pct = self.get_timing_report()
        print('Runtimes:')
        for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
            print('%s: %0.2f%% (%s)' % (label, pct, pretty_time_delta(sec)))

    def update_stat_book(self, stat_book: StatBook, gp_models: List[GPModel]) -> None:
        """Update model population statistics.

        :param stat_book:
        :param gp_models:
        :return:
        """
        stat_book.update_stat_book(data=gp_models, x=self.x_train, base_kernels=self.grammar.base_kernel_names,
                                   n_dims=self.n_dims)

    @classmethod
    def boms_experiment(cls, dataset, **kwargs):
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        base_kernel_names = CKSGrammar.get_base_kernel_names(n_dims)
        hyperpriors = boms_hyperpriors()
        grammar = BOMSGrammar(base_kernel_names, n_dims, hyperpriors)
        kernel_selector = BOMS_kernel_selector(n_parents=1)
        objective = log_likelihood_normalized
        init_qs = BOMSInitQueryStrategy()
        acq = ExpectedImprovementPerSec()
        qs = BestScoreStrategy(scoring_func=acq)
        # kernel = hellinger_kernel_kernel(x_train)
        return cls(grammar, kernel_selector, objective, x_train, y_train, x_test, y_test, eval_budget=50,
                   init_query_strat=init_qs, query_strat=qs, use_surrogate=True, use_laplace=True, **kwargs)

    @classmethod
    def cks_experiment(cls, dataset, **kwargs):
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        grammar = CKSGrammar(n_dims)
        kernel_selector = CKS_kernel_selector(n_parents=1)

        def negative_BIC(m):
            """Computes the negative of the Bayesian Information Criterion (BIC)."""
            return -BIC(m)

        # Use the negative BIC because we want to maximize the objective.
        objective = negative_BIC

        # use conjugate gradient descent for CKS
        optimizer = 'scg'
        return cls(grammar, kernel_selector, objective, x_train, y_train, x_test, y_test, max_depth=10,
                   optimizer=optimizer, use_surrogate=False, use_laplace=False, **kwargs)

    @classmethod
    def evolutionary_experiment(cls,
                                dataset,
                                **kwargs):
        # x_train, x_test, y_train, y_test = dataset.split_train_test()
        x, y = dataset.load_or_generate_data()
        n_dims = x.shape[1]
        base_kernels_names = CKSGrammar.get_base_kernel_names(n_dims)

        pop_size = 25
        m_prob = 0.10
        cx_prob = 0.60
        variation_pct = m_prob + cx_prob  # 60% of individuals created using crossover and 10% mutation
        n_offspring = int(variation_pct * pop_size)
        n_parents = n_offspring

        mutator = HalfAndHalfMutator(operands=[k for k in get_all_1d_kernels(base_kernels_names, n_dims)],
                                     binary_tree_node_cls=KernelNode, max_depth=1)
        recombinator = SubtreeExchangeLeafBiasedRecombinator()
        cx_variator = CrossoverVariator(recombinator, n_offspring=n_offspring, c_prob=cx_prob)
        mut_variator = MutationVariator(mutator, m_prob=m_prob)
        variators = [cx_variator, mut_variator]
        pop_operator = CrossMutPopOperator(variators)
        grammar = EvolutionaryGrammar(base_kernel_names=base_kernels_names, n_dims=n_dims,
                                      population_operator=pop_operator)
        initializer = HalfAndHalfGenerator(binary_operators=grammar.operators, max_depth=1, operands=mutator.operands)
        grammar.initializer = initializer
        grammar.n_init_trees = pop_size

        kernel_selector = evolutionary_kernel_selector(n_parents=n_parents, max_offspring=pop_size)
        objective = log_likelihood_normalized
        budget = 50
        return cls(grammar, kernel_selector, objective, x, y, x_test=None, y_test=None,
                   tabu_search=False, eval_budget=budget, max_null_queries=budget, max_same_expansions=budget,
                   use_surrogate=False, **kwargs)

    @classmethod
    def random_experiment(cls,
                          dataset,
                          **kwargs):
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        grammar = RandomGrammar(n_dims)
        objective = log_likelihood_normalized
        kernel_selector = CKS_kernel_selector(n_parents=1)
        return cls(grammar, kernel_selector, objective, x_train, y_train, x_test, y_test, eval_budget=50,
                   use_surrogate=False, tabu_search=False, **kwargs)

    def update_object_time_predictor(self,
                                     n_hyperparams: int,
                                     time: float):
        self.n_kernel_params.append(n_hyperparams)
        self.objective_times.append(time)

    def create_hellinger_db(self, active_models: ActiveSet, data_X):
        """Create Hellinger distance builder with all active models being candidates"""
        # todo: get this from boms hyperpriors
        lik_noise_std = np.log(0.01)
        lik_noise_mean = 1
        noise_prior = Gaussian(lik_noise_std, lik_noise_mean)

        initial_model_indices = active_models.get_candidate_indices()

        num_samples = 20
        max_num_hyperparameters = 40
        max_num_kernels = 1000

        builder = HellingerDistanceBuilder(noise_prior, num_samples, max_num_hyperparameters, max_num_kernels,
                                           active_models, initial_model_indices, data_X=data_X)

        return builder


# stats functions
def get_model_scores(gp_models: List[GPModel], *args, **kwargs) -> List[float]:
    return [gp_model.score for gp_model in gp_models if gp_model.evaluated]


def get_n_operands(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    return [n_base_kernels(gp_model.covariance.raw_kernel) for gp_model in gp_models]


def get_n_hyperparams(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    return [gp_model.covariance.raw_kernel.size for gp_model in gp_models]


def get_cov_dists(gp_models: List[GPModel], *args, **kwargs) -> Union[np.ndarray, List[int]]:
    kernels = [gp_model.covariance for gp_model in gp_models]
    if len(kernels) >= 2:
        x = kwargs.get('x')
        return pairwise_centered_alignments(kernels, x)
    else:
        return [0] * len(gp_models)


def get_diversity_scores(gp_models: List[GPModel], *args, **kwargs) -> Union[float, List[int]]:
    kernels = [gp_model.covariance for gp_model in gp_models]
    if len(kernels) >= 2:
        base_kernels = kwargs.get('base_kernels')
        n_dims = kwargs.get('n_dims')
        return all_pairs_avg_dist(kernels, base_kernels, n_dims)
    else:
        return [0] * len(gp_models)


def get_best_n_operands(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    model_scores = get_model_scores(gp_models, *args, **kwargs)
    n_operands = get_n_operands(gp_models)
    score_arg_max = int(np.argmax(model_scores))
    return [n_operands[score_arg_max]]


def get_best_n_hyperparams(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
    model_scores = get_model_scores(gp_models, *args, **kwargs)
    n_hyperparams = get_n_hyperparams(gp_models, *args, **kwargs)
    score_arg_max = int(np.argmax(model_scores))
    return [n_hyperparams[score_arg_max]]


def base_kern_freq(base_kern: str) -> Callable[[List[GPModel], Any, Any], List[int]]:
    def get_frequency(gp_models: List[GPModel], *args, **kwargs) -> List[int]:
        cls = KERNEL_DICT[base_kern]
        return [type_count(gp_model.covariance.to_binary_tree(), cls) for gp_model in gp_models]

    return get_frequency

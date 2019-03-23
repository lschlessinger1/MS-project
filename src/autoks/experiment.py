import warnings
from time import time
from typing import Callable, List, Tuple, Optional, FrozenSet, Dict

import matplotlib.pyplot as plt
import numpy as np
from GPy.core import GP
from GPy.core.parameterization.priors import Prior
from GPy.models import GPRegression
from numpy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler

from src.autoks.grammar import BaseGrammar
from src.autoks.kernel import n_base_kernels, covariance_distance, remove_duplicate_aks_kernels, all_pairs_avg_dist, \
    AKSKernel, pretty_print_aks_kernels, kernel_to_infix, sort_kernel, set_priors
from src.autoks.kernel_selection import KernelSelector
from src.autoks.model import set_model_kern, is_nan_model, log_likelihood_normalized, AIC, BIC, pl2
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_svr, rmse_lin_reg, rmse_rbf, rmse_knn, \
    ExperimentReportGenerator
from src.autoks.query_strategy import NaiveQueryStrategy, QueryStrategy
from src.autoks.statistics import StatBookCollection, Statistic, StatBook
from src.evalg.plotting import plot_best_so_far, plot_distribution


class Experiment:
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
    hyperpriors: Optional[Dict[str, Dict[str, Prior]]]
    init_query_strat: Optional[QueryStrategy]
    query_strat: Optional[QueryStrategy]
    gp_model: Optional[GP]
    additive_form: bool
    debug: bool
    verbose: bool
    optimizer: Optional[str]
    n_restarts_optimizer: int

    def __init__(self, grammar, kernel_selector, objective, kernel_families, x_train, y_train, x_test, y_test,
                 standardize_x=True, standardize_y=True, eval_budget=50, max_depth=None, gp_model=None,
                 init_query_strat=None, query_strat=None, hyperpriors=None, additive_form=False, debug=False,
                 verbose=False, optimizer=None, n_restarts_optimizer=10):
        self.grammar = grammar
        self.kernel_selector = kernel_selector
        self.objective = objective
        self.kernel_families = kernel_families

        self.x_train = np.atleast_2d(x_train)
        self.x_test = np.atleast_2d(x_test)
        # Make y >= 2-dimensional
        self.y_train = np.atleast_2d(y_train)
        self.y_test = np.atleast_2d(y_test)
        # only save scaled version of data
        if standardize_x or standardize_y:
            scaler = StandardScaler()
            if standardize_x:
                self.x_train = scaler.fit_transform(self.x_train)
                self.x_test = scaler.transform(self.x_test)
            if standardize_y:
                self.y_train = scaler.fit_transform(self.y_train)
                self.y_test = scaler.transform(self.y_test)
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
        n_hyperparams_name = 'n_hyperparameters'
        n_operands_name = 'n_operands'
        score_name = 'score'
        cov_dists_name = 'cov_dists'
        diversity_scores_name = 'diversity_scores'
        best_stat_name = 'best'
        shared_multi_stat_names = [n_hyperparams_name, n_operands_name]  # All stat books track these variables

        # raw value statistics
        shared_stats = [get_n_hyperparams, get_n_operands]

        evaluations_name = 'evaluations'
        active_set_name = 'active_set'
        expansion_name = 'expansion'
        stat_book_names = [evaluations_name, expansion_name, active_set_name]
        self.stat_book_collection = StatBookCollection(stat_book_names, shared_multi_stat_names, shared_stats)

        sb_active_set = self.stat_book_collection.stat_books['active_set']
        sb_active_set.add_raw_value_stat(score_name, get_model_scores)
        sb_active_set.add_raw_value_stat(cov_dists_name, get_cov_dists)
        sb_active_set.add_raw_value_stat(diversity_scores_name, get_diversity_scores)
        sb_active_set.multi_stats[n_hyperparams_name].add_statistic(Statistic(best_stat_name, get_best_n_hyperparams))
        sb_active_set.multi_stats[n_operands_name].add_statistic(Statistic(best_stat_name, get_best_n_operands))

        sb_evals = self.stat_book_collection.stat_books['evaluations']
        sb_evals.add_raw_value_stat(score_name, get_model_scores)
        # sb_evals.multi_stats[n_hyperparams_name].add_statistic(Statistic(best_stat_name, get_best_n_hyperparams))
        # sb_evals.multi_stats[n_operands_name].add_statistic(Statistic(best_stat_name, get_best_n_operands))

        self.total_optimization_time = 0
        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_kernel_search_time = 0
        self.total_query_time = 0

        self.hyperpriors = hyperpriors

        if gp_model is not None:
            # do not use hyperpriors if gp model is given.
            self.gp_model = gp_model
        else:
            # default model is GP Regression
            self.gp_model = GPRegression(self.x_train, self.y_train)
            if self.hyperpriors is not None:
                # set likelihood hyperpriors
                likelihood_priors = self.hyperpriors['GP']
                self.gp_model.likelihood = set_priors(self.gp_model.likelihood, likelihood_priors)
            # randomize likilihood
            self.gp_model.likelihood.randomize()

        if init_query_strat is not None:
            self.init_query_strat = init_query_strat
        else:
            self.init_query_strat = NaiveQueryStrategy()

        if query_strat is not None:
            self.query_strat = query_strat
        else:
            self.query_strat = NaiveQueryStrategy()

        # self.x_axis_evals = x_axis_evals

    def kernel_search(self) -> List[AKSKernel]:
        """Perform automated kernel search.

        :return: list of kernels
        """
        t_init = time()

        # initialize models
        kernels = self.grammar.initialize(self.kernel_families, n_dims=self.n_dims, hyperpriors=self.hyperpriors)
        kernels = self.randomize_kernels(kernels, verbose=self.verbose)

        # convert to additive form if necessary
        if self.additive_form:
            for aks_kernel in kernels:
                aks_kernel.to_additive_form()

        # Select kernels by acquisition function to be evaluated
        selected_kernels, ind, acq_scores = self.query_kernels(kernels, self.init_query_strat, self.hyperpriors)
        unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
        unselected_kernels = [unevaluated_kernels[i] for i in range(len(unevaluated_kernels)) if i not in ind]
        newly_evaluated_kernels = self.opt_and_eval_kernels(selected_kernels)
        evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated and kernel not in selected_kernels]
        kernels = newly_evaluated_kernels + unselected_kernels + evaluated_kernels
        self.update_stat_book(self.stat_book_collection.stat_books['active_set'], kernels)

        max_same_expansions = 3  # Maximum number of same kernel proposal before terminating
        max_null_queries = 3  # Maximum number of empty queries in a row allowed before terminating
        prev_expansions = []
        prev_n_queried = []
        depth = 0
        while self.n_evals < self.eval_budget:
            if depth > self.max_depth:
                break

            if self.debug and depth % 2 == 0:
                if self.max_depth < np.inf:
                    print('Starting iteration %d/%d' % (depth, self.max_depth))
                print('Evaluated %d/%d kernels' % (self.n_evals, self.eval_budget))

            parents = self.select_parents(kernels)

            # Fix each parent before expansion for use in and initialization optimization, skipping nan-evaluated
            # kernels
            for parent in self.remove_nan_scored_kernels(parents):
                parent.kernel.fix()

            new_kernels = self.propose_new_kernels(parents)
            self.update_stat_book(self.stat_book_collection.stat_books['expansion'], new_kernels)
            # Check for same expansions
            if self.all_same_expansion(new_kernels, prev_expansions, max_same_expansions):
                if self.verbose:
                    print(f'Terminating kernel search. The last {max_same_expansions} expansions proposed the same '
                          f'kernels.')
                break
            else:
                prev_expansions = self.update_kernel_infix_set(new_kernels, prev_expansions, max_same_expansions)

            new_kernels = self.randomize_kernels(new_kernels, verbose=self.verbose)
            kernels += new_kernels

            # evaluate, prune, and optimize kernels
            n_before = len(kernels)
            kernels = remove_duplicate_aks_kernels(kernels)
            if self.verbose:
                n_removed = n_before - len(kernels)
                print(f'Removed {n_removed} duplicate kernels.')

            # Select kernels by acquisition function to be evaluated
            selected_kernels, ind, acq_scores = self.query_kernels(kernels, self.query_strat, self.hyperpriors)

            # Check for empty queries
            prev_n_queried.append(len(ind))
            if all([n == 0 for n in prev_n_queried[-max_null_queries:]]) and len(prev_n_queried) >= max_null_queries:
                if self.verbose:
                    print(f'Terminating kernel search. The last {max_null_queries} queries were empty.')
                break

            unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
            unselected_kernels = [unevaluated_kernels[i] for i in range(len(unevaluated_kernels)) if i not in ind]
            newly_evaluated_kernels = self.opt_and_eval_kernels(selected_kernels)
            evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated and kernel not in selected_kernels]
            kernels = newly_evaluated_kernels + unselected_kernels + evaluated_kernels

            kernels = self.prune_kernels(kernels, acq_scores, ind)
            kernels = self.select_offspring(kernels)
            self.update_stat_book(self.stat_book_collection.stat_books['active_set'], kernels)
            depth += 1

        self.total_kernel_search_time += time() - t_init

        return kernels

    def query_kernels(self,
                      kernels: List[AKSKernel],
                      query_strategy: QueryStrategy,
                      hyperpriors: Optional[Dict[str, Dict[str, Prior]]] = None) \
            -> Tuple[List[AKSKernel], List[int], List[float]]:
        """Select kernels using the acquisition function of the query strategy.

        :param kernels:
        :param query_strategy:
        :param hyperpriors:
        :return:
        """
        t0 = time()
        unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]
        ind, acq_scores = query_strategy.query(unevaluated_kernels, self.x_train, self.y_train, hyperpriors)
        selected_kernels = query_strategy.select(np.array(unevaluated_kernels), acq_scores)
        self.total_query_time += time() - t0

        if self.verbose:
            n_selected = len(ind)
            plural_suffix = '' if n_selected == 1 else 's'
            print(f'Query strategy selected {n_selected} kernel{plural_suffix}:')

            acq_scores_selected = [s for i, s in enumerate(acq_scores) if i in ind]
            for kern, score in zip(selected_kernels, acq_scores_selected):
                print(str(kern), 'score:', score)

        return selected_kernels, ind, acq_scores

    def select_parents(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Choose parents to later expand.

        :param kernels:
        :return:
        """
        evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
        kernel_scores = [kernel.score for kernel in evaluated_kernels]
        parents = self.kernel_selector.select_parents(evaluated_kernels, kernel_scores)
        # Print parent (seed) kernels
        if self.debug:
            pretty_print_aks_kernels(parents, 'Parent')

        return parents

    def propose_new_kernels(self, parents: List[AKSKernel]) -> List[AKSKernel]:
        """Propose new kernels using the grammar given a list of parent kernels.

        :param parents:
        :return:
        """
        t0_exp = time()
        new_kernels = self.grammar.expand(parents, self.kernel_families, self.n_dims, verbose=self.verbose,
                                          hyperpriors=self.hyperpriors)
        self.total_expansion_time += time() - t0_exp

        if self.additive_form:
            for aks_kernel in new_kernels:
                aks_kernel.to_additive_form()

        return new_kernels

    def prune_kernels(self,
                      kernels: List[AKSKernel],
                      acq_scores: List[float],
                      ind: List[int]) -> List[AKSKernel]:
        """Remove un-evaluated kernels if necessary.

        :param kernels:
        :param acq_scores:
        :param ind:
        :return:
        """
        evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
        unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated]

        # acquisition scores of un-evaluated kernels
        acq_scores_unevaluated = [s for i, s in enumerate(acq_scores) if i not in ind]

        pruned_candidates = self.kernel_selector.prune_candidates(unevaluated_kernels, acq_scores_unevaluated)
        kernels = evaluated_kernels + pruned_candidates

        if self.verbose:
            n_before_prune = len(unevaluated_kernels)
            n_pruned = n_before_prune - len(pruned_candidates)
            print(f'Kernel pruner removed {n_pruned}/{n_before_prune} un-evaluated kernels')

        return kernels

    def select_offspring(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Select next round of kernels.

        :param kernels:
        :return:
        """
        # scores = [kernel.score for kernel in kernels]

        # Prioritize keeping evaluated models.
        # TODO: Check correctness
        augmented_scores = [k.score if k.evaluated and not k.nan_scored else -np.inf for k in kernels]

        offspring = self.kernel_selector.select_offspring(kernels, augmented_scores)

        if self.verbose:
            print(f'Offspring selector kept {len(kernels)}/{len(offspring)} kernels')

        return offspring

    def opt_and_eval_kernels(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Optimize and evaluate all kernels

        :param kernels:
        :return:
        """
        evaluated_kernels = []

        for aks_kernel in kernels:
            t0 = time()

            optimized_kernel = self.optimize_kernel(aks_kernel)

            t1 = time()
            self.total_optimization_time += t1 - t0

            evaluated_kernel = self.evaluate_kernel(optimized_kernel)

            self.total_eval_time += time() - t1
            evaluated_kernels.append(evaluated_kernel)

            if not evaluated_kernel.nan_scored:
                self.update_stat_book(self.stat_book_collection.stat_books['evaluations'], [evaluated_kernel])

        evaluated_kernels = self.remove_nan_scored_kernels(evaluated_kernels)

        if self.verbose:
            print('Printing all results')
            # Sort kernels by scores with un-evaluated kernels last
            for k in sorted(evaluated_kernels, key=lambda x: (x.score is not None, x.score), reverse=True):
                print(str(k), 'score:', k.score)

        return evaluated_kernels

    def optimize_kernel(self, aks_kernel: AKSKernel) -> AKSKernel:
        """Optimize the hyperparameters of the kernel

        All of the parameters which were part of the previous parent parent kernel are initialized to their previous
        values. All parameterized are then optimized, randomly restarting the newly introduced parameters.

        :param aks_kernel:
        :return:
        """
        if not aks_kernel.evaluated:
            try:
                kernel = aks_kernel.kernel
                k_unfixed = kernel.copy()
                k_unfixed.unfix()

                # Optimize unfixed kernel (all params unfixed)
                set_model_kern(self.gp_model, k_unfixed)
                self.gp_model.optimize(ipython_notebook=False, optimizer=self.optimizer)

                # Optimize (with restarts) the newly added parameters
                k_fixed = kernel

                # Set param values of fixed kernel to the previously optimized ones of unfixed kernel
                new_params = k_unfixed.param_array
                k_fixed[:] = new_params

                set_model_kern(self.gp_model, k_fixed)
                self.gp_model.optimize_restarts(ipython_notebook=False, optimizer=self.optimizer,
                                                num_restarts=self.n_restarts_optimizer, verbose=False)

                # Unfix all params and set kernel
                k_fixed.unfix()
                set_model_kern(self.gp_model, k_fixed)
                aks_kernel.kernel = self.gp_model.kern
                aks_kernel.lik_params = self.gp_model.likelihood[:].copy()
            except LinAlgError:
                warnings.warn('Y covariance of kernel %s is not positive semi-definite' % aks_kernel)
        return aks_kernel

    def evaluate_kernel(self, aks_kernel: AKSKernel) -> AKSKernel:
        """Evaluate a given kernel using the objective function

        :param aks_kernel:
        :return:
        """
        if not aks_kernel.evaluated:
            set_model_kern(self.gp_model, aks_kernel.kernel)
            self.gp_model.likelihood[:] = aks_kernel.lik_params

            # Check if parameters are well-defined:
            aks_kernel.nan_scored = is_nan_model(self.gp_model)
            if not aks_kernel.nan_scored:
                score = self.objective(self.gp_model)
                self.n_evals += 1
                # aks_kernel.evaluated = True
                aks_kernel.score = score
            else:
                aks_kernel.score = np.nan
                # also count a nan-evaluated kernel as an evaluation
                self.n_evals += 1
        return aks_kernel

    def all_same_expansion(self, new_kernels: List[AKSKernel],
                           prev_expansions: List[FrozenSet[str]],
                           max_expansions: int) -> bool:
        kernels_infix_new = self.kernel_to_infix_set(new_kernels)
        all_same = all([s == kernels_infix_new for s in prev_expansions])
        return all_same and len(prev_expansions) == max_expansions

    def kernel_to_infix_set(self, aks_kernels: List[AKSKernel]) -> FrozenSet[str]:
        kernels_sorted = [sort_kernel(aks_kernel.kernel) for aks_kernel in aks_kernels]
        return frozenset([kernel_to_infix(kernel) for kernel in kernels_sorted])

    def update_kernel_infix_set(self, new_kernels: List[AKSKernel],
                                prev_expansions: List[FrozenSet[str]],
                                max_expansions: int) -> List[FrozenSet[str]]:
        expansions = prev_expansions.copy()
        if len(prev_expansions) == max_expansions:
            expansions = expansions[1:]
        elif len(prev_expansions) < max_expansions:
            expansions += [self.kernel_to_infix_set(new_kernels)]

        return expansions

    @staticmethod
    def randomize_kernels(aks_kernels: List[AKSKernel],
                          verbose: bool = False) -> List[AKSKernel]:
        if verbose:
            print('Randomizing kernels')

        for aks_kernel in aks_kernels:
            aks_kernel.kernel.randomize()

        return aks_kernels

    def remove_nan_scored_kernels(self, aks_kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Remove all kernels that have NaN scores.

        :param aks_kernels:
        :return:
        """
        return [aks_kernel for aks_kernel in aks_kernels if not aks_kernel.nan_scored]

    def summarize(self, aks_kernels: List[AKSKernel]) -> None:
        """Summarize the experiment.

        :param aks_kernels:
        :return:
        """
        evaluated_kernels = [kernel for kernel in aks_kernels if kernel.evaluated]
        sorted_aks_kernels = sorted(evaluated_kernels, key=lambda x: x.score, reverse=True)
        best_aks_kernel = sorted_aks_kernels[0]
        best_kernel = best_aks_kernel.kernel
        best_model = self.gp_model.__class__(self.x_train, self.y_train, kernel=best_kernel)
        # Set the likelihood parameters.
        best_model.likelihood[:] = best_aks_kernel.lik_params

        # If training data is 1D, show a plot.
        if best_model.input_dim == 1:
            best_model.plot(plot_density=True)
            plt.show()

        # View results of experiment
        for stat_book in self.stat_book_collection.stat_book_list():
            self.plot_stat_book(stat_book)

        print('')
        self.timing_report()
        print('')

        print('Best kernel:')
        best_aks_kernel.pretty_print()

        print('In full form:')
        best_aks_kernel.print_full()
        print('')

        # Summarize model
        nll = -best_model.log_likelihood()
        nll_norm = log_likelihood_normalized(best_model)
        mean_nlpd = np.mean(-best_model.log_predictive_density(self.x_test, self.y_test))
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
            **kwargs) -> List[AKSKernel]:
        """Run the kernel search and optionally summarize and create a report

        :param summarize:
        :param create_report:
        :param kwargs:
        :return:
        """
        aks_kernels = self.kernel_search()
        if summarize:
            self.summarize(aks_kernels)

        if create_report:
            if self.verbose:
                print('')
                print('Creating report...')
            report_gen = ExperimentReportGenerator(self, aks_kernels, self.x_test, self.y_test)
            report_gen.summarize_experiment(**kwargs)

        return aks_kernels

    def plot_stat_book(self, stat_book: StatBook):
        ms = stat_book.multi_stats
        if 'score' in ms:
            self.plot_best_scores(stat_book)
            self.plot_score_summary(stat_book)
        if 'n_hyperparameters' in ms:
            self.plot_n_hyperparams_summary(stat_book)
        if 'n_operands' in ms:
            self.plot_n_operands_summary(stat_book)
        if 'cov_dists' in ms:
            self.plot_cov_dist_summary(stat_book)
        if 'diversity_scores' in ms:
            self.plot_kernel_diversity_summary(stat_book)

    def plot_best_scores(self, stat_book: StatBook) -> None:
        """Plot the best models scores

        :return:
        """
        if stat_book.name == 'evaluations':
            best_scores = stat_book.running_max('score')
            x_label = ' evaluations'
        else:
            best_scores = stat_book.maximum('score')
            x_label = 'generation'

        plot_best_so_far(best_scores, x_label=x_label)
        plt.show()

    def plot_score_summary(self, stat_book: StatBook) -> None:
        """Plot a summary of model scores

        :return:
        """
        if stat_book.name == 'evaluations':
            best_scores = stat_book.running_max('score')
            mean_scores = stat_book.running_mean('score')
            std_scores = stat_book.running_std('score')
            x_label = 'evaluations'
        else:
            best_scores = stat_book.maximum('score')
            mean_scores = stat_book.mean('score')
            std_scores = stat_book.std('score')
            x_label = 'generation'

        plot_distribution(mean_scores, std_scores, best_scores, x_label=x_label)
        plt.show()

    def plot_n_hyperparams_summary(self, stat_book: StatBook) -> None:
        """Plot a summary of the number of hyperparameters

        :return:
        """
        x_label = 'evaluations' if stat_book.name == 'evaluations' else 'generation'
        if 'best' in stat_book.multi_stats['n_hyperparameters'].stats:
            best_n_hyperparameters = stat_book.multi_stats['n_hyperparameters'].stats['best'].data
        else:
            best_n_hyperparameters = None
        median_n_hyperparameters = stat_book.median('n_hyperparameters')
        std_n_hyperparameters = stat_book.std('n_hyperparameters')
        plot_distribution(median_n_hyperparameters, std_n_hyperparameters, best_n_hyperparameters,
                          value_name='median', metric_name=stat_book.name+'# Hyperparameters', x_label=x_label)
        plt.show()

    def plot_n_operands_summary(self, stat_book: StatBook) -> None:
        """Plot a summary of the number of operands

        :return:
        """
        x_label = 'evaluations' if stat_book.name == 'evaluations' else 'generation'
        if 'best' in stat_book.multi_stats['n_operands'].stats:
            best_n_operands = stat_book.multi_stats['n_operands'].stats['best'].data
        else:
            best_n_operands = None
        median_n_operands = stat_book.median('n_operands')
        std_n_operands = stat_book.std('n_operands')
        plot_distribution(median_n_operands, std_n_operands, best_n_operands, value_name='median',
                          metric_name='# Operands', x_label=x_label)
        plt.show()

    def plot_cov_dist_summary(self, stat_book: StatBook) -> None:
        """Plot a summary of the homogeneity of models over each generation.

        :return:
        """
        x_label = 'evaluations' if stat_book.name == 'evaluations' else 'generation'
        mean_cov_dists = stat_book.mean('cov_dists')
        std_cov_dists = stat_book.std('cov_dists')
        plot_distribution(mean_cov_dists, std_cov_dists, metric_name='covariance distance', x_label=x_label)
        plt.show()

    def plot_kernel_diversity_summary(self, stat_book: StatBook) -> None:
        """Plot a summary of the diversity of models over each generation.

        :return:
        """
        x_label = 'evaluations' if stat_book.name == 'evaluations' else 'generation'
        mean_diversity_scores = stat_book.mean('diversity_scores')
        std_diversity_scores = stat_book.std('diversity_scores')
        plot_distribution(mean_diversity_scores, std_diversity_scores, metric_name='diversity',
                          value_name='population', x_label=x_label)
        plt.show()

    def get_timing_report(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get the runtime report of the kernel search.

        :return:
        """
        eval_time = self.total_eval_time
        opt_time = self.total_optimization_time
        expansion_time = self.total_expansion_time
        query_time = self.total_query_time
        total_time = self.total_kernel_search_time
        other_time = total_time - eval_time - opt_time - expansion_time - query_time

        labels = ['Evaluation', 'Optimization', 'Expansion', 'Query', 'Other']
        x = np.array([eval_time, opt_time, expansion_time, query_time, other_time])
        x_pct = 100 * (x / total_time)

        return labels, x, x_pct

    def timing_report(self) -> None:
        """Print a runtime report of the kernel search.

        :return:
        """
        labels, x, x_pct = self.get_timing_report()
        print('Runtimes:')
        for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
            print('%s: %0.2f%% (%0.2fs)' % (label, pct, sec))

    def update_stat_book(self, stat_book: StatBook, aks_kernels: List[AKSKernel]) -> None:
        """Update kernel population statistics.

        :param stat_book:
        :param aks_kernels:
        :return:
        """
        stat_book.update_stat_book(data=aks_kernels, x=self.x_train, base_kernels=self.kernel_families,
                                   n_dims=self.n_dims)


# stats functions
def get_model_scores(aks_kernels: List[AKSKernel], *args, **kwargs):
    return [aks_kernel.score for aks_kernel in aks_kernels if aks_kernel.evaluated]


def get_n_operands(aks_kernels: List[AKSKernel], *args, **kwargs):
    return [n_base_kernels(aks_kernel.kernel) for aks_kernel in aks_kernels]


def get_n_hyperparams(aks_kernels: List[AKSKernel], *args, **kwargs):
    return [aks_kernel.kernel.param_array.size for aks_kernel in aks_kernels]


def get_cov_dists(aks_kernels: List[AKSKernel], *args, **kwargs):
    kernels = [aks_kernel.kernel for aks_kernel in aks_kernels]
    if len(kernels) >= 2:
        x = kwargs.get('x')
        return covariance_distance(kernels, x)
    else:
        return [0] * len(aks_kernels)


def get_diversity_scores(aks_kernels: List[AKSKernel], *args, **kwargs):
    kernels = [aks_kernel.kernel for aks_kernel in aks_kernels]
    if len(kernels) >= 2:
        base_kernels = kwargs.get('base_kernels')
        n_dims = kwargs.get('n_dims')
        return all_pairs_avg_dist(kernels, base_kernels, n_dims)
    else:
        return [0] * len(aks_kernels)


def get_best_n_operands(aks_kernels: List[AKSKernel], *args, **kwargs):
    model_scores = get_model_scores(aks_kernels, *args, **kwargs)
    n_operands = get_n_operands(aks_kernels)
    score_arg_max = int(np.argmax(model_scores))
    return [n_operands[score_arg_max]]


def get_best_n_hyperparams(aks_kernels: List[AKSKernel], *args, **kwargs):
    model_scores = get_model_scores(aks_kernels, *args, **kwargs)
    n_hyperparams = get_n_hyperparams(aks_kernels, *args, **kwargs)
    score_arg_max = int(np.argmax(model_scores))
    return [n_hyperparams[score_arg_max]]

import warnings
from time import time
from typing import List, Tuple, Optional, Callable, Union, Any

import numpy as np
from GPy.core import GP
from GPy.inference.latent_function_inference import Laplace
from GPy.models import GPRegression
from numpy.linalg import LinAlgError

from src.autoks.backend.kernel import set_priors, n_base_kernels, KERNEL_DICT
from src.autoks.backend.model import is_nan_model, set_model_kern
from src.autoks.core.covariance import pairwise_centered_alignments, all_pairs_avg_dist
from src.autoks.core.gp_model import GPModel, remove_nan_scored_models, pretty_print_gp_models, \
    remove_duplicate_gp_models, randomize_models, update_kernel_infix_set, all_same_expansion
from src.autoks.core.grammar import BaseGrammar
from src.autoks.core.hyperprior import Hyperpriors
from src.autoks.core.kernel_selection import KernelSelector
from src.autoks.core.query_strategy import QueryStrategy, NaiveQueryStrategy
from src.autoks.statistics import StatBook, StatBookCollection, Statistic
from src.autoks.util import type_count


class ModelSelector:
    """Abstract base class for model selectors"""
    selected_models: List[GPModel]
    grammar: BaseGrammar
    kernel_selector: KernelSelector
    objective: Callable[[GP], float]
    kernel_families: List[str]
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

    def __init__(self, standardize_y, grammar, kernel_selector, objective, eval_budget=50,
                 max_depth=None, gp_model=None, init_query_strat=None, query_strat=None, additive_form=False,
                 debug=False,
                 verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3, optimizer=None,
                 n_restarts_optimizer=10, use_laplace=True):
        self.grammar = grammar
        self.kernel_selector = kernel_selector
        self.objective = objective

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

        self.total_optimization_time = 0
        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_model_search_time = 0
        self.total_query_time = 0

        if gp_model is not None:
            # do not use hyperpriors if gp model is given.
            self.gp_model = gp_model

        self.standardize_y = standardize_y
        self.use_laplace = use_laplace

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

        # Used for expected improvement per second.
        self.n_kernel_params = []
        self.objective_times = []

        # visited set of all expanded kernel expressions previously evaluated
        self.visited = set()

        # statistics used for plotting
        base_kernel_names = self.grammar.base_kernel_names
        self.n_hyperparams_name = 'n_hyperparameters'
        self.n_operands_name = 'n_operands'
        self.base_kern_freq_names = [base_kern_name + '_frequency' for base_kern_name in base_kernel_names]
        self.score_name = 'score'
        self.cov_dists_name = 'cov_dists'
        self.diversity_scores_name = 'diversity_scores'
        self.best_stat_name = 'best'
        # All stat books track these variables
        shared_multi_stat_names = [self.n_hyperparams_name, self.n_operands_name] + self.base_kern_freq_names

        # raw value statistics
        base_kern_stat_funcs = [base_kern_freq(base_kern_name) for base_kern_name in base_kernel_names]
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

    def train(self, x, y):
        self.gp_model = GPRegression(x, y, normalizer=self.standardize_y)
        # default model is GP Regression
        if self.use_laplace:
            self.gp_model.inference_method = Laplace()

        if self.grammar.hyperpriors is not None:
            # set likelihood hyperpriors
            likelihood_priors = self.grammar.hyperpriors['GP']
            self.gp_model.likelihood = set_priors(self.gp_model.likelihood, likelihood_priors)
        # randomize likelihood
        self.gp_model.likelihood.randomize()

        # set selected models
        t_init = time()

        # initialize models
        kernels = self.grammar.initialize()
        kernels = remove_duplicate_gp_models(kernels)  # TODO: clean up
        kernels = randomize_models(kernels)

        # convert to additive form if necessary
        if self.additive_form:
            for gp_model in kernels:
                gp_model.covariance.to_additive_form()

        # Select gp_models by acquisition function to be evaluated
        selected_kernels, ind, acq_scores = self.query_models(kernels, self.init_query_strat, x, y,
                                                              self.grammar.hyperpriors)
        kernels = self.train_models(kernels, selected_kernels, ind)

        self.update_stat_book(self.stat_book_collection.stat_books[self.active_set_name], kernels, x)

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
            for parent in remove_nan_scored_models(parents):
                parent.covariance.raw_kernel.fix()

            new_kernels = self.propose_new_kernels(parents)
            self.update_stat_book(self.stat_book_collection.stat_books[self.expansion_name], new_kernels, x)
            # Check for same expansions
            if all_same_expansion(new_kernels, prev_expansions, self.max_same_expansions):
                if self.verbose:
                    print(f'Terminating kernel search. The last {self.max_same_expansions} expansions proposed the same'
                          f' gp_models.')
                break
            else:
                prev_expansions = update_kernel_infix_set(new_kernels, prev_expansions, self.max_same_expansions)

            new_kernels = randomize_models(new_kernels)
            kernels += new_kernels

            # evaluate, prune, and optimize gp_models
            n_before = len(kernels)
            kernels = remove_duplicate_gp_models(kernels)
            if self.verbose:
                n_removed = n_before - len(kernels)
                print(f'Removed {n_removed} duplicate gp_models.\n')

            # Select gp_models by acquisition function to be evaluated
            selected_kernels, ind, acq_scores = self.query_models(kernels, self.query_strat, x, y,
                                                                  self.grammar.hyperpriors)

            # Check for empty queries
            prev_n_queried.append(len(ind))
            if all([n == 0 for n in prev_n_queried[-self.max_null_queries:]]) and \
                    len(prev_n_queried) >= self.max_null_queries:
                if self.verbose:
                    print(f'Terminating kernel search. The last {self.max_null_queries} queries were empty.')
                break

            kernels = self.train_models(kernels, selected_kernels, ind)

            kernels = self.select_offspring(kernels)
            self.update_stat_book(self.stat_book_collection.stat_books[self.active_set_name], kernels, x)
            depth += 1

        self.total_model_search_time += time() - t_init

        self.selected_models = kernels
        return self

    def predict(self, x):
        # for now, just use best model to predict. Should be using model averaging...
        # sort selected models by scores
        pass

    def score(self, x, y) -> float:
        pass

    def initialize(self):
        raise NotImplementedError

    def print_search_summary(self, depth, kernels):
        print(f'Iteration {depth}/{self.max_depth}')
        print(f'Evaluated {self.n_evals}/{self.eval_budget}')
        evaluated_gp_models = [gp_model for gp_model in remove_nan_scored_models(kernels)
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

    def query_models(self,
                     kernels: List[GPModel],
                     query_strategy: QueryStrategy,
                     x_train,
                     y_train,
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
        ind, acq_scores = query_strategy.query(unevaluated_kernels_ind, kernels, x_train, y_train,
                                               hyperpriors, None, durations=self.objective_times,
                                               n_hyperparams=self.n_kernel_params)
        selected_kernels = query_strategy.select(np.array(unevaluated_kernels), np.array(acq_scores))
        self.total_query_time += time() - t0

        if self.verbose:
            n_selected = len(ind)
            plural_suffix = '' if n_selected == 1 else 's'
            print(f'Query strategy selected {n_selected} kernel{plural_suffix}:')

            acq_scores_selected = [s for i, s in enumerate(acq_scores) if i in ind]
            for kern, score in zip(selected_kernels, acq_scores_selected):
                kern.covariance.pretty_print()
                print('\tacq. score =', score)
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
        # Prioritize keeping evaluated models.
        augmented_scores = [k.score if k.evaluated and not k.nan_scored else -np.inf for k in kernels]

        offspring = self.kernel_selector.select_offspring(kernels, augmented_scores)

        if self.verbose:
            print(f'Offspring selector kept {len(offspring)}/{len(kernels)} gp_models\n')

        return offspring

    def train_models(self,
                     all_models: List[GPModel],
                     chosen_models: List[GPModel],
                     indices: List[int]):
        unevaluated_kernels = [kernel for kernel in all_models if not kernel.evaluated]
        unselected_kernels = [unevaluated_kernels[i] for i in range(len(unevaluated_kernels)) if i not in indices]
        newly_evaluated_kernels = self.opt_and_eval_models(chosen_models)
        for gp_model in newly_evaluated_kernels:
            self.visited.add(gp_model.covariance.symbolic_expr_expanded)
        old_evaluated_kernels = [kernel for kernel in all_models if
                                 kernel.evaluated and kernel not in chosen_models]
        return newly_evaluated_kernels + unselected_kernels + old_evaluated_kernels

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
                self.update_stat_book(self.stat_book_collection.stat_books[self.evaluations_name], [evaluated_model],
                                      x_train=None)

        evaluated_models = remove_nan_scored_models(evaluated_models)

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
                gp_model.score = score
            else:
                gp_model.score = np.nan

        return gp_model

    def best_model(self) -> GPModel:
        """Get the best scoring model."""
        evaluated_gp_models = [model for model in self.selected_models if model.evaluated]
        sorted_gp_models = sorted(evaluated_gp_models, key=lambda x: x.score, reverse=True)
        return sorted_gp_models[0]

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

    def update_stat_book(self,
                         stat_book: StatBook,
                         gp_models: List[GPModel],
                         x_train) -> None:
        """Update model population statistics.

        :param stat_book:
        :param gp_models:
        :return:
        """
        stat_book.update_stat_book(data=gp_models, x=x_train, base_kernels=self.grammar.base_kernel_names,
                                   n_dims=self.grammar.n_dims)

    def update_object_time_predictor(self,
                                     n_hyperparams: int,
                                     time: float):
        self.n_kernel_params.append(n_hyperparams)
        self.objective_times.append(time)


class EvolutionaryModelSelector(ModelSelector):

    def __init__(self, standardize_y, grammar, kernel_selector, objective, eval_budget=50,
                 max_depth=None, gp_model=None, init_query_strat=None, query_strat=None, additive_form=False,
                 debug=False, verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3,
                 optimizer=None, n_restarts_optimizer=10, use_laplace=True):
        super().__init__(standardize_y, grammar, kernel_selector, objective, eval_budget, max_depth,
                         gp_model, init_query_strat, query_strat, additive_form, debug, verbose, tabu_search,
                         max_null_queries, max_same_expansions, optimizer, n_restarts_optimizer, use_laplace)

    def initialize(self):
        pass


class BomsModelSelector(ModelSelector):
    def __init__(self, standardize_y, grammar, kernel_selector, objective, eval_budget=50,
                 max_depth=None, gp_model=None, init_query_strat=None, query_strat=None, additive_form=False,
                 debug=False, verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3,
                 optimizer=None, n_restarts_optimizer=10, use_laplace=True):
        super().__init__(standardize_y, grammar, kernel_selector, objective, eval_budget, max_depth,
                         gp_model, init_query_strat, query_strat, additive_form, debug, verbose, tabu_search,
                         max_null_queries, max_same_expansions, optimizer, n_restarts_optimizer, use_laplace)

    def initialize(self):
        pass


class CKSModelSelector(ModelSelector):
    def __init__(self, standardize_y, grammar, kernel_selector, objective, eval_budget=50,
                 max_depth=None, gp_model=None, init_query_strat=None, query_strat=None, additive_form=False,
                 debug=False, verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3,
                 optimizer=None, n_restarts_optimizer=10, use_laplace=True):
        super().__init__(standardize_y, grammar, kernel_selector, objective, eval_budget, max_depth,
                         gp_model, init_query_strat, query_strat, additive_form, debug, verbose, tabu_search,
                         max_null_queries, max_same_expansions, optimizer, n_restarts_optimizer, use_laplace)

    def initialize(self):
        pass


class RandomModelSelector(ModelSelector):

    def __init__(self, standardize_y, grammar, kernel_selector, objective, eval_budget=50,
                 max_depth=None, gp_model=None, init_query_strat=None, query_strat=None, additive_form=False,
                 debug=False, verbose=False, tabu_search=True, max_null_queries=3, max_same_expansions=3,
                 optimizer=None, n_restarts_optimizer=10, use_laplace=True):
        super().__init__(standardize_y, grammar, kernel_selector, objective, eval_budget, max_depth,
                         gp_model, init_query_strat, query_strat, additive_form, debug, verbose, tabu_search,
                         max_null_queries, max_same_expansions, optimizer, n_restarts_optimizer, use_laplace)

    def initialize(self):
        pass


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

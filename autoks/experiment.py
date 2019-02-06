from time import time

import numpy as np
from GPy.models import GPRegression
from numpy.linalg import LinAlgError

from autoks.grammar import BaseGrammar
from autoks.kernel import n_base_kernels, covariance_distance, remove_duplicate_aks_kernels, all_pairs_avg_dist
from autoks.model import set_model_kern
from evalg.plotting import plot_best_so_far, plot_distribution


class Experiment:
    grammar: BaseGrammar

    def __init__(self, grammar, objective, kernel_families, X, y, eval_budget=50, max_depth=10, gp_model=None,
                 debug=False, verbose=False, optimizer=None, n_restarts_optimizer=10):
        self.grammar = grammar
        self.objective = objective
        self.kernel_families = kernel_families
        self.X = X
        self.y = y
        self.n_dims = self.X.shape[1]
        # number of model evaluations (budget)
        self.eval_budget = eval_budget
        self.max_depth = max_depth
        self.n_evals = 0
        self.debug = debug
        self.verbose = verbose
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.n_init_kernels = 15

        # statistics used for plotting
        self.best_scores = []
        self.mean_scores = []
        self.std_scores = []
        self.median_n_hyperparameters = []
        self.std_n_hyperparameters = []
        self.best_n_hyperparameters = []
        self.median_n_operands = []
        self.std_n_operands = []
        self.best_n_operands = []
        self.mean_cov_dists = []
        self.std_cov_dists = []
        self.diversity_scores = []
        self.total_optimization_time = 0
        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_kernel_search_time = 0

        if gp_model is not None:
            self.gp_model = gp_model
        else:
            # default model is GP Regression
            self.gp_model = GPRegression(self.X, self.y)

    def kernel_search(self):
        """ Perform automated kernel search

        :return: list of kernels
        """
        t_init = time()
        # initialize models
        kernels = self.grammar.initialize(self.kernel_families, n_kernels=self.n_init_kernels, n_dims=self.n_dims)

        self.opt_and_eval_kernels(kernels)

        depth = 0
        while self.n_evals < self.eval_budget:
            if depth > self.max_depth:
                break

            if self.debug and depth % 2 == 0:
                print('Done iteration %d/%d' % (depth, self.max_depth))
                print('Evaluated %d/%d kernels' % (self.n_evals, self.eval_budget))
                print('Best Score: %.2f' % max([k.score for k in kernels]))

            # Get next round of kernels
            parents = self.grammar.select_parents(np.array(kernels)).tolist()

            # Print parent (seed) kernels
            if self.debug:
                n_parents = len(parents)
                print('Best (%d) kernel%s:' % (n_parents, 's' if n_parents > 1 else ''))
                for parent in parents:
                    parent.pretty_print()

            # fix each parent before expansion for use in optimization
            for parent in parents:
                parent.kernel.fix()

            t0_exp = time()
            new_kernels = self.grammar.expand(parents, self.kernel_families, self.n_dims, verbose=self.verbose)
            self.total_expansion_time += time() - t0_exp

            kernels += new_kernels

            # evaluate, prune, and optimize kernels
            kernels = remove_duplicate_aks_kernels(kernels)

            self.opt_and_eval_kernels(kernels)

            if self.verbose:
                print('Printing all results')
                for k in sorted(kernels, key=lambda k: k.score, reverse=True):
                    print(str(k), 'score:', k.score)

            # Select next round of kernels
            kernels = self.grammar.select_offspring(np.array(kernels)).tolist()

            depth += 1

        self.total_kernel_search_time += time() - t_init
        return kernels

    def opt_and_eval_kernels(self, kernels):
        """ Optimize and evaluate kernels

        :param kernels:
        :return:
        """
        for aks_kernel in kernels:
            t0 = time()

            self.optimize_kernel(aks_kernel)

            t1 = time()
            self.total_optimization_time += t1 - t0

            self.evaluate_kernel(aks_kernel)

            self.total_eval_time += time() - t1

        self.update_stats(kernels)

    def optimize_kernel(self, aks_kernel):
        if not aks_kernel.scored:
            try:
                kernel = aks_kernel.kernel
                k_copy = kernel.copy()
                k_copy.unfix()

                # Optimize k_copy (with all params unfixed)
                set_model_kern(self.gp_model, k_copy)
                self.gp_model.optimize(ipython_notebook=False, optimizer=self.optimizer)

                # Optimize (with restarts) the newly added parameters
                k_new = kernel

                # Set param values of k_new to the previously optimized ones of k_copy
                new_params = k_copy.param_array
                k_new[:] = new_params

                set_model_kern(self.gp_model, k_new)
                self.gp_model.optimize_restarts(ipython_notebook=False, optimizer=self.optimizer,
                                                num_restarts=self.n_restarts_optimizer, verbose=False)

                # Unfix all params and set kernel
                k_new.unfix()
                set_model_kern(self.gp_model, k_new)
                aks_kernel.kernel = self.gp_model.kern
            except LinAlgError:
                print('Y covariance is not positive semi-definite')

    def evaluate_kernel(self, aks_kernel):
        if not aks_kernel.scored:
            set_model_kern(self.gp_model, aks_kernel.kernel)
            score = self.objective(self.gp_model)
            self.n_evals += 1
            aks_kernel.scored = True
            aks_kernel.score = score

    def plot_best_scores(self):
        """ Plot the best models scores

        :return:
        """
        plot_best_so_far(self.best_scores)

    def plot_score_summary(self):
        """ Plot a summary of model scores

        :return:
        """
        plot_distribution(self.mean_scores, self.std_scores, self.best_scores)

    def plot_n_hyperparams_summary(self):
        """ Plot a summary of the number of hyperparameters

        :return:
        """
        plot_distribution(self.median_n_hyperparameters, self.std_n_hyperparameters, self.best_n_hyperparameters,
                          value_name='median', metric_name='# Hyperparameters')

    def plot_n_operands_summary(self):
        """ Plot a summary of the number of operands

        :return:
        """
        plot_distribution(self.median_n_operands, self.std_n_operands, self.best_n_operands, value_name='median',
                          metric_name='# Operands')

    def plot_cov_dist_summary(self):
        """Plot a summary of the homogeneity of models over each generation."""
        plot_distribution(self.mean_cov_dists, self.std_cov_dists, metric_name='covariance distance')

    def plot_kernel_diversity_summary(self):
        """Plot a summary of the diversity of models over each generation."""
        plot_distribution(self.diversity_scores, metric_name='diversity', value_name='population')

    def timing_report(self):
        """Print a runtime report of the kernel search."""
        eval_time = self.total_eval_time
        opt_time = self.total_optimization_time
        expansion_time = self.total_expansion_time
        total_time = self.total_kernel_search_time
        other_time = total_time - eval_time - opt_time - expansion_time
        labels = ['Evaluation', 'Optimization', 'Expansion', 'Other']
        x = np.array([eval_time, opt_time, expansion_time, other_time])

        x_pct = 100 * (x / total_time)
        print('Runtimes:')
        for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
            print('%s: %0.2f%% (%0.2fs)' % (label, pct, sec))

    def update_stats(self, kernels):
        """ Update kernel population statistics

        :param kernels:
        :return:
        """
        model_scores = np.array([k.score for k in kernels])
        score_argmax = np.argmax(model_scores)
        self.best_scores.append(model_scores[score_argmax])
        self.mean_scores.append(np.mean(model_scores))
        self.std_scores.append(np.std(model_scores))

        n_params = np.array([aks_kernel.kernel.param_array.size for aks_kernel in kernels])
        self.median_n_hyperparameters.append(np.median(n_params))
        self.std_n_hyperparameters.append(np.std(n_params))
        self.best_n_hyperparameters.append(n_params[score_argmax])

        n_operands = np.array([n_base_kernels(aks_kernel.kernel) for aks_kernel in kernels])
        self.median_n_operands.append(np.median(n_operands))
        self.std_n_operands.append(np.std(n_operands))
        self.best_n_operands.append(n_operands[score_argmax])

        cov_dists = covariance_distance([aks_kernel.kernel for aks_kernel in kernels], self.X)
        self.mean_cov_dists.append(np.mean(cov_dists))
        self.std_cov_dists.append(np.std(cov_dists))

        diversity_score = all_pairs_avg_dist([aks_kernel.kernel for aks_kernel in kernels], self.kernel_families,
                                             self.n_dims)
        self.diversity_scores.append(diversity_score)

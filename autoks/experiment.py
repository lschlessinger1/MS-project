import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
from GPy.models import GPRegression
from numpy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler

from autoks.grammar import BaseGrammar
from autoks.kernel import n_base_kernels, covariance_distance, remove_duplicate_aks_kernels, all_pairs_avg_dist
from autoks.model import set_model_kern, is_nan_model, log_likelihood_normalized, AIC, BIC, pl2
from autoks.postprocessing import ExperimentReportGenerator, compute_gpy_model_rmse, rmse_svr, rmse_lin_reg, rmse_rbf, \
    rmse_knn
from evalg.plotting import plot_best_so_far, plot_distribution


class Experiment:
    grammar: BaseGrammar

    def __init__(self, grammar, objective, kernel_families, X_train, y_train, X_test, y_test, standardize_X=True,
                 standardize_y=True, eval_budget=50, max_depth=10, gp_model=None, debug=False, verbose=False,
                 optimizer=None, n_restarts_optimizer=10):
        self.grammar = grammar
        self.objective = objective
        self.kernel_families = kernel_families

        self.X_train = X_train
        self.X_test = X_test
        # Make y >= 2-dimensional
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        self.y_test = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
        # only save scaled version of data
        if standardize_X or standardize_y:
            scaler = StandardScaler()
            if standardize_X:
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
            if standardize_y:
                self.y_train = scaler.fit_transform(self.y_train)
                self.X_test = scaler.transform(self.y_test)
        self.n_dims = self.X_train.shape[1]

        self.eval_budget = eval_budget  # number of model evaluations (budget)
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
            self.gp_model = GPRegression(self.X_train, self.y_train)

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
                print('Starting iteration %d/%d' % (depth, self.max_depth))
                print('Evaluated %d/%d kernels' % (self.n_evals, self.eval_budget))

            # Get next round of kernels
            parents = self.grammar.select_parents(np.array(kernels)).tolist()

            # Print parent (seed) kernels
            if self.debug:
                n_parents = len(parents)
                print('Best (%d) kernel%s:' % (n_parents, 's' if n_parents > 1 else ''))
                for parent in parents:
                    parent.pretty_print()

            # Fix each parent before expansion for use in optimization, skipping nan-scored kernels
            for parent in self.remove_nan_scored_kernels(parents):
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
                for k in sorted(kernels, key=lambda x: x.score, reverse=True):
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
        # kernels = self.remove_nan_scored_kernels(kernels)

        for aks_kernel in kernels:
            t0 = time()

            self.optimize_kernel(aks_kernel)

            t1 = time()
            self.total_optimization_time += t1 - t0

            self.evaluate_kernel(aks_kernel)

            self.total_eval_time += time() - t1

        kernels = self.remove_nan_scored_kernels(kernels)
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
                warnings.warn('Y covariance of kernel %s is not positive semi-definite' % aks_kernel)

    def evaluate_kernel(self, aks_kernel):
        if not aks_kernel.scored:
            set_model_kern(self.gp_model, aks_kernel.kernel)

            # Check if parameters are well-defined:
            aks_kernel.nan_scored = is_nan_model(self.gp_model)
            if not aks_kernel.nan_scored:
                score = self.objective(self.gp_model)
                self.n_evals += 1
                # aks_kernel.scored = True
                aks_kernel.score = score
            else:
                aks_kernel.score = np.nan
                # also count a nan-scored kernel as an evaluation
                self.n_evals += 1

    def remove_nan_scored_kernels(self, aks_kernels):
        return [aks_kernel for aks_kernel in aks_kernels if not aks_kernel.nan_scored]

    def summarize(self, aks_kernels):
        sorted_aks_kernels = sorted(aks_kernels, key=lambda x: x.score, reverse=True)
        best_aks_kernel = sorted_aks_kernels[0]
        best_kernel = best_aks_kernel.kernel
        best_model = self.gp_model.__class__(self.X_train, self.y_train, kernel=best_kernel)

        # If training data is 1D, show a plot.
        if best_model.input_dim == 1:
            best_model.plot(plot_density=True)
            plt.show()

        # View results of experiment
        self.plot_best_scores()
        self.plot_score_summary()
        self.plot_n_hyperparams_summary()
        self.plot_n_operands_summary()
        self.plot_cov_dist_summary()
        self.plot_kernel_diversity_summary()

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
        mean_nlpd = np.mean(-best_model.log_predictive_density(self.X_test, self.y_test))
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
        best_model_rmse = compute_gpy_model_rmse(best_model, self.X_test, self.y_test)
        svm_rmse = rmse_svr(self.X_train, self.y_train, self.X_test, self.y_test)
        lr_rmse = rmse_lin_reg(self.X_train, self.y_train, self.X_test, self.y_test)
        se_rmse = rmse_rbf(self.X_train, self.y_train, self.X_test, self.y_test)
        knn_rmse = rmse_knn(self.X_train, self.y_train, self.X_test, self.y_test)

        print('RMSE Best Model = %.3f' % best_model_rmse)
        print('RMSE Linear Regression = %.3f' % lr_rmse)
        print('RMSE SVM = %.3f' % svm_rmse)
        print('RMSE RBF = %.3f' % se_rmse)
        print('RMSE k-NN = %.3f' % knn_rmse)

    def run(self, summarize=True, create_report=True):
        aks_kernels = self.kernel_search()
        if summarize:
            self.summarize(aks_kernels)

        if create_report:
            if self.verbose:
                print('')
                print('Creating report...')
            report_gen = ExperimentReportGenerator(self, aks_kernels, self.X_test, self.y_test)
            report_gen.summarize_experiment()

        return aks_kernels

    def plot_best_scores(self):
        """ Plot the best models scores

        :return:
        """
        plot_best_so_far(self.best_scores)
        plt.show()

    def plot_score_summary(self):
        """ Plot a summary of model scores

        :return:
        """
        plot_distribution(self.mean_scores, self.std_scores, self.best_scores)
        plt.show()

    def plot_n_hyperparams_summary(self):
        """ Plot a summary of the number of hyperparameters

        :return:
        """
        plot_distribution(self.median_n_hyperparameters, self.std_n_hyperparameters, self.best_n_hyperparameters,
                          value_name='median', metric_name='# Hyperparameters')
        plt.show()

    def plot_n_operands_summary(self):
        """ Plot a summary of the number of operands

        :return:
        """
        plot_distribution(self.median_n_operands, self.std_n_operands, self.best_n_operands, value_name='median',
                          metric_name='# Operands')
        plt.show()

    def plot_cov_dist_summary(self):
        """Plot a summary of the homogeneity of models over each generation."""
        plot_distribution(self.mean_cov_dists, self.std_cov_dists, metric_name='covariance distance')
        plt.show()

    def plot_kernel_diversity_summary(self):
        """Plot a summary of the diversity of models over each generation."""
        plot_distribution(self.diversity_scores, metric_name='diversity', value_name='population')
        plt.show()

    def get_timing_report(self):
        eval_time = self.total_eval_time
        opt_time = self.total_optimization_time
        expansion_time = self.total_expansion_time
        total_time = self.total_kernel_search_time
        other_time = total_time - eval_time - opt_time - expansion_time

        labels = ['Evaluation', 'Optimization', 'Expansion', 'Other']
        x = np.array([eval_time, opt_time, expansion_time, other_time])
        x_pct = 100 * (x / total_time)

        return labels, x, x_pct

    def timing_report(self):
        """Print a runtime report of the kernel search."""
        labels, x, x_pct = self.get_timing_report()
        print('Runtimes:')
        for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
            print('%s: %0.2f%% (%0.2fs)' % (label, pct, sec))

    def update_stats(self, kernels):
        """ Update kernel population statistics

        :param kernels:
        :return:
        """
        if len(kernels) > 0:
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

            cov_dists = covariance_distance([aks_kernel.kernel for aks_kernel in kernels], self.X_train)
            self.mean_cov_dists.append(np.mean(cov_dists))
            self.std_cov_dists.append(np.std(cov_dists))

            diversity_score = all_pairs_avg_dist([aks_kernel.kernel for aks_kernel in kernels], self.kernel_families,
                                                 self.n_dims)
            self.diversity_scores.append(diversity_score)

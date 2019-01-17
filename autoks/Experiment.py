import numpy as np
from GPy.models import GPRegression
from numpy.linalg import LinAlgError

from autoks.grammar import BaseGrammar, remove_duplicate_aks_kernels
from autoks.model import set_model_kern
from evalg.plotting import plot_best_so_far, plot_score_summary


class Experiment:
    grammar: BaseGrammar

    def __init__(self, grammar, objective, kernel_families, X, y, eval_budget=50, max_depth=10, gp_model=None,
                 debug=False, verbose=False, optimizer=None):
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
        self.n_init_kernels = 15

        # statistics used for plotting
        self.best_scores = []
        self.mean_scores = []
        self.std_scores = []

        if gp_model is not None:
            self.gp_model = gp_model
        else:
            # default model is GP Regression
            self.gp_model = GPRegression(self.X, self.y)

    def kernel_search(self):
        """ Perform automated kernel search

        :return: list of kernels
        """
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

            new_kernels = self.grammar.expand(parents, self.kernel_families, self.n_dims, verbose=self.verbose)
            kernels += new_kernels

            # evaluate, prune, and optimize kernels
            kernels = remove_duplicate_aks_kernels(kernels)

            self.opt_and_eval_kernels(kernels)

            if self.verbose:
                print('Printing all results')
                for k in kernels:
                    print(str(k), 'score:', k.score)

            # Select next round of kernels
            kernels = self.grammar.select_offspring(np.array(kernels)).tolist()

            depth += 1

        return kernels

    def opt_and_eval_kernels(self, kernels):
        """ Optimize and evaluate kernels

        :param kernels:
        :return:
        """
        for aks_kernel in kernels:
            self.optimize_kernel(aks_kernel)
            self.evaluate_kernel(aks_kernel)

        self.update_stats([k.score for k in kernels])

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
                self.gp_model.optimize_restarts(ipython_notebook=False, optimizer=self.optimizer, verbose=False)

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
        plot_score_summary(self.mean_scores, self.std_scores, self.best_scores)

    def update_stats(self, model_scores):
        """ Update model score statistics

        :param model_scores:
        :return:
        """
        self.best_scores.append(max(model_scores))
        self.mean_scores.append(np.mean(model_scores))
        self.std_scores.append(np.std(model_scores))

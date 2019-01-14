import numpy as np
from GPy.models import GPRegression
from numpy.linalg import LinAlgError

from autoks.grammar import BaseGrammar, remove_duplicate_aks_kernels
from autoks.kernel import set_model_kern
from evalg.plotting import plot_best_so_far, plot_score_summary


class Experiment:
    grammar: BaseGrammar

    def __init__(self, grammar, objective, kernel_families, X, y, eval_budget=50, gp_model=None, debug=False):
        self.grammar = grammar
        self.objective = objective
        self.kernel_families = kernel_families
        self.X = X
        self.y = y
        self.n_dims = self.X.shape[1]
        # number of model evaluations (budget)
        self.eval_budget = eval_budget
        self.n_evals = 0
        self.debug = debug
        self.n_init_kernels = 15

        # if plotting
        self.best_scores = []
        self.mean_scores = []
        self.std_scores = []

        # default model is GP Regression
        if gp_model is not None:
            self.gp_model = gp_model
        else:
            # self.gp_model = GPRegression
            self.gp_model = GPRegression(self.X, self.y)

    def kernel_search(self):
        """ Perform automated kernel search

        :return:
        """
        # initialize models
        kernels = self.grammar.initialize(self.kernel_families, n_kernels=self.n_init_kernels, n_dims=self.n_dims)

        kernels = self.optimize_kernels(kernels)
        model_scores = self.evaluate_kernels(kernels)

        # if plotting
        self.update_stats(model_scores)

        for i in range(self.eval_budget):
            if self.debug and i % (self.eval_budget // 10) == 0:
                print('Done iteration %d/%d' % (i, self.eval_budget))
                print('Evaluated %d kernels' % self.n_evals)
                print('Best Score: %.2f' % max(model_scores))

            # Get next round of kernels
            new_kernels = self.grammar.expand(kernels, model_scores, self.kernel_families, self.n_dims)
            kernels += new_kernels

            # evaluate, prune, and optimize kernels
            kernels = remove_duplicate_aks_kernels(kernels)
            kernels = self.optimize_kernels(kernels)
            model_scores = self.evaluate_kernels(kernels)

            # Select next round of kernels
            kernels = self.grammar.select(np.array(kernels), model_scores).tolist()

            # if plotting
            self.update_stats(model_scores)

        return kernels, model_scores

    def evaluate_kernels(self, kernels):
        """ Calculate fitness/objective for all models

        :param kernels:
        :return:
        """
        scores = []
        for aks_kernel in kernels:
            if not aks_kernel.scored:
                # model = self.gp_model(self.X, self.y, kernel=aks_kernel.kernel)
                set_model_kern(self.gp_model, aks_kernel.kernel)
                scores.append(self.objective(self.gp_model))
                self.n_evals += 1
                aks_kernel.scored = True
        model_scores = np.array(scores)
        return model_scores

    def optimize_kernels(self, kernels):
        """ Optimize hyperparameters of all kernels

        :param kernels:
        :return:
        """
        kernels_optimized = []
        for aks_kernel in kernels:

            if not aks_kernel.scored:
                try:
                    set_model_kern(self.gp_model, aks_kernel.kernel)
                    self.gp_model.optimize(ipython_notebook=False)
                    aks_kernel.kernel = self.gp_model.kern
                    kernels_optimized.append(aks_kernel)
                except LinAlgError:
                    print('Y covariance is not positive semi-definite')
        return kernels_optimized

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

import numpy as np
from GPy.models import GPRegression
from numpy.linalg import LinAlgError

from autoks.grammar import BaseGrammar
from autoks.kernel import kernel_to_infix
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
        self.n_init_models = 15

        # if plotting
        self.best_scores = []
        self.mean_scores = []
        self.std_scores = []

        # default model is GP Regression
        if gp_model is not None:
            self.gp_model = gp_model
        else:
            self.gp_model = GPRegression

    def kernel_search(self):
        """ Perform automated kernel search

        :return:
        """
        # initialize models
        kernels = self.grammar.initialize(self.kernel_families, n_models=self.n_init_models, n_dims=self.n_dims)
        models = [self.gp_model(self.X, self.y, kernel=kernel) for kernel in kernels]

        model_scores = self.evaluate_models(models)

        # if plotting
        self.update_stats(model_scores)

        for i in range(self.eval_budget):
            if self.debug and i % 10 == 0:
                print('Done iteration %d / %d' % (i, self.eval_budget))
                print('Best Score: %.2f' % max(model_scores))

            # Get next round of models
            new_models = self.grammar.expand(models, model_scores, self.kernel_families)
            models += new_models

            # evaluate, prune, and optimize models
            model_scores = self.evaluate_models(models)
            models = self.optimize_models(models)

            # Select next round of models
            models = self.grammar.select(np.array(models), model_scores).tolist()

            # if plotting
            self.update_stats(model_scores)

        return models, model_scores

    def evaluate_models(self, models):
        """ Calculate fitness/objective for all models

        :param models:
        :return:
        """
        scores = []
        for model in models:
            scores.append(self.objective(self.X, self.y, model))
            self.n_evals += 1
        model_scores = np.array(scores)
        return model_scores

    def optimize_models(self, models):
        """ Optimize hyperparameters of all models

        :param models:
        :return:
        """
        for model in models:
            print(kernel_to_infix(model.kern))
            try:
                model.optimize()
            except LinAlgError:
                print('Y covariance is not positive semi-definite')
        return models

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

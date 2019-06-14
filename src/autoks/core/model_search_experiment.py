import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.autoks.backend.model import log_likelihood_normalized, AIC, BIC, pl2
from src.autoks.core.experiment import BaseExperiment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.model_selection import BomsModelSelector, CKSModelSelector, EvolutionaryModelSelector, \
    RandomModelSelector
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.plotting import plot_kernel_tree, plot_best_scores, plot_score_summary, plot_n_hyperparams_summary, \
    plot_n_operands_summary, plot_base_kernel_freqs, plot_cov_dist_summary, plot_kernel_diversity_summary
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_lin_reg, rmse_svr, rmse_rbf, rmse_knn, rmse_to_smse
from src.autoks.statistics import StatBook
from src.autoks.tracking import ModelSearchTracker
from src.autoks.util import pretty_time_delta
from src.datasets.dataset import Dataset


class ModelSearchExperiment(BaseExperiment):

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 model_selector: ModelSelector,
                 x_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 tracker: ModelSearchTracker = None,
                 hide_warnings: bool = True):
        super().__init__(x_train, y_train, x_test, y_test, hide_warnings=hide_warnings)
        self.model_selector = model_selector
        self.tracker = tracker

    def run(self,
            eval_budget: int = 50,
            max_n_generations: Optional[int] = None,
            verbose: int = 2) -> None:
        """Run the model search experiment"""
        if self.hide_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_selector.train(self.x_train, self.y_train, eval_budget=eval_budget,
                                          max_generations=max_n_generations, tracker=self.tracker, verbose=verbose)
        else:
            self.model_selector.train(self.x_train, self.y_train, eval_budget=eval_budget,
                                      max_generations=max_n_generations, tracker=self.tracker, verbose=verbose)

        self.summarize(self.model_selector.best_model())

    def summarize(self, best_gp_model: GPModel) -> None:
        """Summarize results of experiment"""
        best_model = best_gp_model.build_model(self.x_train, self.y_train)

        # If training data is 1D, show a plot.
        if best_model.input_dim == 1:
            best_model.plot(plot_density=True, title='Best Model')
            plt.show()

        # View results of experiment
        for stat_book in self.tracker.stat_book_collection.stat_book_list():
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
        aic = AIC(best_model)
        bic = BIC(best_model)
        pl2_score = pl2(best_model)

        print('NLL = %.3f' % nll)
        print('NLL (normalized) = %.3f' % nll_norm)
        if self.has_test_data:
            mean_nlpd = np.mean(-best_model.log_predictive_density(self.x_test, self.y_test))
            print('NLPD = %.3f' % mean_nlpd)
        print('AIC = %.3f' % aic)
        print('BIC = %.3f' % bic)
        print('PL2 = %.3f' % pl2_score)
        print('')

        # Compare RMSE of best model to other models
        if self.has_test_data:
            best_model_rmse = compute_gpy_model_rmse(best_model, self.x_test, self.y_test)
            svm_rmse = rmse_svr(self.x_train, self.y_train, self.x_test, self.y_test)
            lr_rmse = rmse_lin_reg(self.x_train, self.y_train, self.x_test, self.y_test)
            se_rmse = rmse_rbf(self.x_train, self.y_train, self.x_test, self.y_test)
            knn_rmse = rmse_knn(self.x_train, self.y_train, self.x_test, self.y_test)

            print('SMSE Best Model = %.3f' % rmse_to_smse(best_model_rmse, self.y_test))
            print('SMSE Linear Regression = %.3f' % rmse_to_smse(lr_rmse, self.y_test))
            print('SMSE SVM = %.3f' % rmse_to_smse(svm_rmse, self.y_test))
            print('SMSE RBF = %.3f' % rmse_to_smse(se_rmse, self.y_test))
            print('SMSE k-NN = %.3f' % rmse_to_smse(knn_rmse, self.y_test))

    def timing_report(self) -> None:
        """Print a runtime report of the model search.

        :return:
        """
        labels, x, x_pct = self.model_selector.get_timing_report()
        print('Runtimes:')
        for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
            print('%s: %0.2f%% (%s)' % (label, pct, pretty_time_delta(sec)))

    def plot_stat_book(self, stat_book: StatBook):
        ms = stat_book.multi_stats
        x_label = 'evaluations' if stat_book.name == self.tracker.evaluations_name else 'generation'
        if self.tracker.score_name in ms:
            plot_best_scores(self.tracker.score_name, self.tracker.evaluations_name, stat_book)
            plot_score_summary(self.tracker.score_name, self.tracker.evaluations_name, stat_book)
        if self.tracker.n_hyperparams_name in ms:
            plot_n_hyperparams_summary(self.tracker.n_hyperparams_name, self.tracker.best_stat_name,
                                       stat_book, x_label)
        if self.tracker.n_operands_name in ms:
            plot_n_operands_summary(self.tracker.n_operands_name, self.tracker.best_stat_name, stat_book,
                                    x_label)
        if all(key in ms for key in self.tracker.base_kern_freq_names):
            plot_base_kernel_freqs(self.tracker.base_kern_freq_names, stat_book, x_label)
        if self.tracker.cov_dists_name in ms:
            plot_cov_dist_summary(self.tracker.cov_dists_name, stat_book, x_label)
        if self.tracker.diversity_scores_name in ms:
            plot_kernel_diversity_summary(self.tracker.diversity_scores_name, stat_book, x_label)

    @classmethod
    def boms_experiment(cls, dataset: Dataset, **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y

        model_selector = BomsModelSelector(**kwargs)

        tracker = ModelSearchTracker()

        return cls(x, y, model_selector, tracker=tracker)

    @classmethod
    def cks_experiment(cls, dataset: Dataset, **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y

        model_selector = CKSModelSelector(fitness_fn=log_likelihood_normalized, optimizer=None, **kwargs)

        tracker = ModelSearchTracker()

        return cls(x, y, model_selector, tracker=tracker)

    @classmethod
    def evolutionary_experiment(cls, dataset: Dataset, **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y

        model_selector = EvolutionaryModelSelector(**kwargs)

        tracker = ModelSearchTracker()

        return cls(x, y, model_selector, tracker=tracker)

    @classmethod
    def random_experiment(cls,
                          dataset: Dataset,
                          **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y

        model_selector = RandomModelSelector(**kwargs)

        tracker = ModelSearchTracker()

        return cls(x, y, model_selector, tracker=tracker)

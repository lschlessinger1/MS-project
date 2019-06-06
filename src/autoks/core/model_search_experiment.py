import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.autoks.backend.kernel import get_all_1d_kernels
from src.autoks.backend.model import log_likelihood_normalized, AIC, BIC, pl2
from src.autoks.core.experiment import BaseExperiment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import CKSGrammar, BOMSGrammar, EvolutionaryGrammar, RandomGrammar
from src.autoks.core.hyperprior import boms_hyperpriors
from src.autoks.core.kernel_encoding import KernelNode
from src.autoks.core.model_selection.boms_model_selector import BomsModelSelector
from src.autoks.core.model_selection.cks_model_selector import CKSModelSelector
from src.autoks.core.model_selection.evolutionary_model_selector import EvolutionaryModelSelector
from src.autoks.core.model_selection.model_selector import ModelSelector
from src.autoks.core.model_selection.random_model_selector import RandomModelSelector
from src.autoks.plotting import plot_kernel_tree, plot_best_scores, plot_score_summary, plot_n_hyperparams_summary, \
    plot_n_operands_summary, plot_base_kernel_freqs, plot_cov_dist_summary, plot_kernel_diversity_summary
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_lin_reg, rmse_svr, rmse_rbf, rmse_knn, rmse_to_smse
from src.autoks.statistics import StatBook
from src.autoks.tracking import ModelSearchTracker
from src.autoks.util import pretty_time_delta
from src.datasets.dataset import Dataset
from src.evalg.genprog import HalfAndHalfMutator, SubtreeExchangeLeafBiasedRecombinator, HalfAndHalfGenerator
from src.evalg.vary import CrossoverVariator, MutationVariator, CrossMutPopOperator


class ModelSearchExperiment(BaseExperiment):

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 model_selector: ModelSelector,
                 x_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 tracker=None,
                 hide_warnings: bool = True):
        super().__init__(x_train, y_train, x_test, y_test, hide_warnings=hide_warnings)
        self.model_selector = model_selector
        self.tracker = tracker

    def run(self) -> None:
        """Run the model search experiment"""
        if self.hide_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_selector.train(self.x_train, self.y_train)
        else:
            self.model_selector.train(self.x_train, self.y_train)

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
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        base_kernel_names = CKSGrammar.get_base_kernel_names(n_dims)
        hyperpriors = boms_hyperpriors()
        grammar = BOMSGrammar(base_kernel_names, n_dims, hyperpriors)

        tracker = ModelSearchTracker(grammar.base_kernel_names)

        model_selector = BomsModelSelector(grammar, use_laplace=True, active_set_callback=tracker.active_set_callback,
                                           eval_callback=tracker.evaluations_callback,
                                           expansion_callback=tracker.expansion_callback, **kwargs)

        return cls(x_train, y_train, model_selector, x_test, y_test, tracker)

    @classmethod
    def cks_experiment(cls, dataset: Dataset, **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y
        n_dims = x.shape[1]
        grammar = CKSGrammar(n_dims)
        tracker = ModelSearchTracker(grammar.base_kernel_names)

        model_selector = CKSModelSelector(grammar, objective=log_likelihood_normalized, max_generations=None,
                                          optimizer=None, active_set_callback=tracker.active_set_callback,
                                          eval_callback=tracker.evaluations_callback,
                                          expansion_callback=tracker.expansion_callback, **kwargs)
        return cls(x, y, model_selector, tracker=tracker)

    @classmethod
    def evolutionary_experiment(cls,
                                dataset: Dataset,
                                **kwargs):
        dataset.load_or_generate_data()
        x, y = dataset.x, dataset.y
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

        tracker = ModelSearchTracker(grammar.base_kernel_names)

        model_selector = EvolutionaryModelSelector(grammar, n_parents=n_parents, max_offspring=pop_size,
                                                   initializer=initializer,
                                                   active_set_callback=tracker.active_set_callback,
                                                   eval_callback=tracker.evaluations_callback,
                                                   expansion_callback=tracker.expansion_callback, **kwargs)
        return cls(x, y, model_selector, tracker=tracker)

    @classmethod
    def random_experiment(cls,
                          dataset: Dataset,
                          **kwargs):
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        grammar = RandomGrammar(n_dims)
        objective = log_likelihood_normalized

        tracker = ModelSearchTracker(grammar.base_kernel_names)

        model_selector = RandomModelSelector(grammar, objective,
                                             active_set_callback=tracker.active_set_callback,
                                             eval_callback=tracker.evaluations_callback,
                                             expansion_callback=tracker.expansion_callback, **kwargs)

        return cls(x_train, y_train, model_selector, x_test, y_test, tracker)

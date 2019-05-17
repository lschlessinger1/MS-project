from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.autoks.backend.kernel import get_all_1d_kernels
from src.autoks.backend.model import log_likelihood_normalized, AIC, BIC, pl2
from src.autoks.core.acquisition_function import ExpectedImprovementPerSec
from src.autoks.core.experiment import BaseExperiment
from src.autoks.core.gp_model import GPModel
from src.autoks.core.grammar import CKSGrammar, BOMSGrammar, EvolutionaryGrammar, RandomGrammar
from src.autoks.core.hyperprior import boms_hyperpriors
from src.autoks.core.kernel_encoding import KernelNode
from src.autoks.core.kernel_selection import BOMS_kernel_selector, CKS_kernel_selector, evolutionary_kernel_selector
from src.autoks.core.model_selection import ModelSelector, EvolutionaryModelSelector, BomsModelSelector, \
    CKSModelSelector, RandomModelSelector
from src.autoks.core.query_strategy import BOMSInitQueryStrategy, BestScoreStrategy
from src.autoks.plotting import plot_kernel_tree, plot_best_scores, plot_score_summary, plot_n_hyperparams_summary, \
    plot_n_operands_summary, plot_base_kernel_freqs, plot_cov_dist_summary, plot_kernel_diversity_summary
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_lin_reg, rmse_svr, rmse_rbf, rmse_knn
from src.autoks.statistics import StatBook
from src.autoks.util import pretty_time_delta
from src.evalg.genprog import HalfAndHalfMutator, SubtreeExchangeLeafBiasedRecombinator, HalfAndHalfGenerator
from src.evalg.vary import CrossoverVariator, MutationVariator, CrossMutPopOperator


class ModelSearchExperiment(BaseExperiment):

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 model_selector: ModelSelector,
                 x_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model_selector = model_selector

    def run(self) -> None:
        """Run the model search experiment"""
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
        for stat_book in self.model_selector.stat_book_collection.stat_book_list():
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
        if self.has_test_data:
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
        if self.has_test_data:
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
        x_label = 'evaluations' if stat_book.name == self.model_selector.evaluations_name else 'generation'
        if self.model_selector.score_name in ms:
            plot_best_scores(self.model_selector.score_name, self.model_selector.evaluations_name, stat_book)
            plot_score_summary(self.model_selector.score_name, self.model_selector.evaluations_name, stat_book)
        if self.model_selector.n_hyperparams_name in ms:
            plot_n_hyperparams_summary(self.model_selector.n_hyperparams_name, self.model_selector.best_stat_name,
                                       stat_book, x_label)
        if self.model_selector.n_operands_name in ms:
            plot_n_operands_summary(self.model_selector.n_operands_name, self.model_selector.best_stat_name, stat_book,
                                    x_label)
        if all(key in ms for key in self.model_selector.base_kern_freq_names):
            plot_base_kernel_freqs(self.model_selector.base_kern_freq_names, stat_book, x_label)
        if self.model_selector.cov_dists_name in ms:
            plot_cov_dist_summary(self.model_selector.cov_dists_name, stat_book, x_label)
        if self.model_selector.diversity_scores_name in ms:
            plot_kernel_diversity_summary(self.model_selector.diversity_scores_name, stat_book, x_label)

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
        model_selector = BomsModelSelector(grammar, kernel_selector, objective, eval_budget=50,
                                           init_query_strat=init_qs, query_strat=qs, use_laplace=True, **kwargs)

        return cls(x_train, y_train, model_selector, x_test, y_test)

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
        model_selector = CKSModelSelector(grammar, kernel_selector, objective, max_depth=10,
                                          optimizer=optimizer, use_laplace=False, **kwargs)
        return cls(x_train, y_train, model_selector, x_test, y_test)

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
        model_selector = EvolutionaryModelSelector(grammar, kernel_selector, objective,
                                                   tabu_search=False, eval_budget=budget, max_null_queries=budget,
                                                   max_same_expansions=budget, **kwargs)
        return cls(x, y, model_selector)

    @classmethod
    def random_experiment(cls,
                          dataset,
                          **kwargs):
        x_train, x_test, y_train, y_test = dataset.split_train_test()
        n_dims = x_train.shape[1]
        grammar = RandomGrammar(n_dims)
        objective = log_likelihood_normalized
        kernel_selector = CKS_kernel_selector(n_parents=1)

        model_selector = RandomModelSelector(grammar, kernel_selector, objective, eval_budget=50,
                                             tabu_search=False, **kwargs)

        return cls(x_train, y_train, model_selector, x_test, y_test, )

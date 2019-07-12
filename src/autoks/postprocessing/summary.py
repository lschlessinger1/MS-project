import argparse
import gzip
import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.autoks.core.fitness_functions import log_likelihood_normalized
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.model_selection_criteria import AIC, BIC, pl2
from src.autoks.plotting import plot_kernel_tree, plot_best_scores, plot_cov_dist_summary, \
    plot_kernel_diversity_summary, plot_base_kernel_freqs, plot_n_operands_summary, plot_n_hyperparams_summary, \
    plot_score_summary
from src.autoks.postprocessing import compute_gpy_model_rmse, rmse_svr, rmse_lin_reg, rmse_rbf, rmse_knn, rmse_to_smse
from src.autoks.statistics import StatBook
from src.autoks.tracking import ModelSearchTracker
from src.autoks.util import pretty_time_delta


def summarize(experiment_dirname):
    """Summarize a single run of an experiment."""
    print(f'Summarizing {experiment_dirname}')
    exp_dict = _parse_experiment(experiment_dirname)

    # Get model selector.
    model_selector_input_dict = exp_dict['model_selector']
    model_selector = ModelSelector.from_dict(model_selector_input_dict)
    print(model_selector)

    # Get best GP model.
    best_gp_model = model_selector.best_model()
    print(best_gp_model)

    # Get tracker.
    tracker = ModelSearchTracker.from_dict(exp_dict["tracker"])
    print(tracker)

    # Get timing report.
    timing_report = model_selector.get_timing_report()
    print(timing_report)

    # Get dataset.
    datasets_module = importlib.import_module('src.datasets')
    dataset_class_ = getattr(datasets_module, exp_dict['dataset_cls'])
    dataset_args = exp_dict.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    x, y = dataset.x, dataset.y
    has_test_data = hasattr(dataset, 'x_test') and hasattr(dataset, 'y_test')
    if has_test_data:
        x_test = dataset.x_test
    else:
        x_test = None

    if has_test_data:
        y_test = dataset.y_test
    else:
        y_test = None

    if model_selector.standardize_x:
        x_scaler = StandardScaler()
        x = x_scaler.fit_transform(x)
        x_test = x_scaler.transform(x_test) if x_test else x_test

    if model_selector.standardize_y:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y)
        y_test = y_scaler.transform(y_test) if y_test else y_test

    best_model = best_gp_model.build_model(x, y)  # assumes no test set

    create_figures(best_gp_model, best_model, tracker)
    print_summary(best_gp_model, best_model, x, y, x_test, y_test)
    print_timing_report(timing_report)


# TODO: use serializer here
def _parse_experiment(experiment_dirname: str) -> dict:
    compress = experiment_dirname.split(".")[-1] == "zip"

    if compress:
        with gzip.GzipFile(experiment_dirname, 'r') as json_data:
            json_bytes = json_data.read()
            json_str = json_bytes.decode('utf-8')
            output_dict = json.loads(json_str)
    else:
        with open(experiment_dirname) as json_data:
            output_dict = json.load(json_data)

    return output_dict


def create_figures(best_gp_model, best_model, tracker):
    # If training data is 1D, show a plot.
    if best_model.input_dim == 1:
        best_model.plot(plot_density=True, title='Best Model')
        plt.show()

    # View results of experiment
    for stat_book in tracker.stat_book_collection.stat_book_list():
        plot_stat_book(tracker, stat_book)

    # Plot the kernel tree of the best model
    plot_kernel_tree(best_gp_model)


def print_summary(best_gp_model, best_model, x_train, y_train, x_test=None, y_test=None):
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

    has_test_data = x_test is not None and y_test is not None
    if has_test_data:
        mean_nlpd = np.mean(-best_model.log_predictive_density(x_test, y_test))
        print('NLPD = %.3f' % mean_nlpd)
    print('AIC = %.3f' % aic)
    print('BIC = %.3f' % bic)
    print('PL2 = %.3f' % pl2_score)
    print('')

    # Compare RMSE of best model to other models
    if has_test_data:
        best_model_rmse = compute_gpy_model_rmse(best_model, x_test, y_test)
        svm_rmse = rmse_svr(x_train, y_train, x_test, y_test)
        lr_rmse = rmse_lin_reg(x_train, y_train, x_test, y_test)
        se_rmse = rmse_rbf(x_train, y_train, x_test, y_test)
        knn_rmse = rmse_knn(x_train, y_train, x_test, y_test)

        print('SMSE Best Model = %.3f' % rmse_to_smse(best_model_rmse, y_test))
        print('SMSE Linear Regression = %.3f' % rmse_to_smse(lr_rmse, y_test))
        print('SMSE SVM = %.3f' % rmse_to_smse(svm_rmse, y_test))
        print('SMSE RBF = %.3f' % rmse_to_smse(se_rmse, y_test))
        print('SMSE k-NN = %.3f' % rmse_to_smse(knn_rmse, y_test))


def print_timing_report(timing_report) -> None:
    """Print a runtime report of the model search.

    :return:
    """
    labels, x, x_pct = timing_report
    print('Runtimes:')
    for pct, sec, label in sorted(zip(x_pct, x, labels), key=lambda v: v[1], reverse=True):
        print('%s: %0.2f%% (%s)' % (label, pct, pretty_time_delta(sec)))


def plot_stat_book(tracker, stat_book: StatBook):
    ms = stat_book.multi_stats
    x_label = 'evaluations' if stat_book.name == tracker.evaluations_name else 'generation'
    if tracker.score_name in ms:
        plot_best_scores(tracker.score_name, tracker.evaluations_name, stat_book)
        plot_score_summary(tracker.score_name, tracker.evaluations_name, stat_book)
    if tracker.n_hyperparams_name in ms:
        plot_n_hyperparams_summary(tracker.n_hyperparams_name, tracker.best_stat_name, stat_book, x_label)
    if tracker.n_operands_name in ms:
        plot_n_operands_summary(tracker.n_operands_name, tracker.best_stat_name, stat_book, x_label)
    if all(key in ms for key in tracker.base_kern_freq_names) and len(tracker.base_kern_freq_names) > 0:
        plot_base_kernel_freqs(tracker.base_kern_freq_names, stat_book, x_label)
    if tracker.cov_dists_name in ms:
        plot_cov_dist_summary(tracker.cov_dists_name, stat_book, x_label)
    if tracker.diversity_scores_name in ms:
        plot_kernel_diversity_summary(tracker.diversity_scores_name, stat_book, x_label)


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_group_dirname", type=str, help="Directory name of an experiment group to run "
                                                                   "postprocessing on.")
    args = parser.parse_args()
    return args


def main():
    """Summarize experiment."""
    args = _parse_args()
    summarize(args.experiment_group_dirname)


if __name__ == '__main__':
    main()

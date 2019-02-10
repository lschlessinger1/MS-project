import datetime
import os

import numpy as np
from pylatex import Document, Section, Figure, NoEscape, SubFigure, Center, Tabu, MiniPage, LineBreak, VerticalSpace, \
    Subsection, Command
from pylatex.utils import bold, italic
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from autoks.kernel import kernel_to_infix
from autoks.model import AIC, BIC, pl2, log_likelihood_normalized
from evalg.plotting import plot_distribution, plot_best_so_far


def compute_skmodel_rmse(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def compute_gpy_model_rmse(model, X_test, y_test):
    mean, var = model.predict(X_test)
    y_pred = mean
    return np.sqrt(mean_squared_error(y_test, y_pred))


def rmse_svr(X_train, y_train, X_test, y_test):
    return compute_skmodel_rmse(SVR(kernel='rbf'), X_train, y_train, X_test, y_test)


def rmse_lin_reg(X_train, y_train, X_test, y_test):
    return compute_skmodel_rmse(LinearRegression(), X_train, y_train, X_test, y_test)


class ExperimentReportGenerator:
    def __init__(self, experiment, aks_kernels, X_test, y_test, results_dir_name='results'):
        self.experiment = experiment
        self.aks_kernels = aks_kernels
        self.results_dir_name = results_dir_name

        self.X_train = self.experiment.X
        self.y_train = self.experiment.y
        self.X_test = X_test
        self.y_test = y_test

        sorted_aks_kernels = sorted(self.aks_kernels, key=lambda x: x.score, reverse=True)
        self.best_aks_kernel = sorted_aks_kernels[0]
        self.best_kernel = self.best_aks_kernel.kernel
        self.best_model = self.experiment.gp_model.__class__(self.experiment.X, self.experiment.y,
                                                             kernel=self.best_kernel)

    def add_overview(self, doc):
        with doc.create(Section('Overview')):
            doc.append('Overview of kernel search results.')
            doc.append("\n")
            doc.append(VerticalSpace("10pt"))
            doc.append(LineBreak())

            best_kern_short = str(self.best_aks_kernel)
            best_kern_long = kernel_to_infix(self.best_aks_kernel.kernel, show_params=True)

            with doc.create(MiniPage()):
                doc.append(bold("Best Kernel:"))
                doc.append("\n")
                doc.append(VerticalSpace("1pt"))
                doc.append(LineBreak())
                doc.append(italic("Short Form:"))
                doc.append("\n")
                doc.append(best_kern_short)
                doc.append("\n")
                doc.append(VerticalSpace("2.5pt"))
                doc.append(LineBreak())
                doc.append(italic("Long Form:"))
                doc.append("\n")
                doc.append(best_kern_long)

    def add_model_scores(self, doc, width, *args, **kwargs):
        with doc.create(Subsection('Model Score Evolution')):
            doc.append('A summary of the distribution of model scores.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_best_so_far(self.experiment.best_scores)
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('I am a caption.')
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.mean_scores, self.experiment.std_scores,
                                      self.experiment.best_scores)
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('I am a caption.')
                plot.add_caption('I am a caption.')

    def add_kernel_structure_subsection(self, doc, width, *args, **kwargs):
        with doc.create(Subsection('Kernel Structure Evolution')):
            doc.append('A summary of the structure of kernels searched.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_distribution(self.experiment.median_n_hyperparameters, self.experiment.std_n_hyperparameters,
                                      self.experiment.best_n_hyperparameters, value_name='median',
                                      metric_name='# Hyperparameters')
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('I am a caption.')
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.median_n_operands,
                                      self.experiment.std_n_operands, self.experiment.best_n_operands,
                                      value_name='median', metric_name='# Operands')
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('I am a caption.')
                plot.add_caption('I am a caption.')

    def add_population_subsection(self, doc, width, *args, **kwargs):
        with doc.create(Subsection('Population Evolution')):
            doc.append('A summary of the population of kernels searched.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_distribution(self.experiment.mean_cov_dists, self.experiment.std_cov_dists,
                                      metric_name='covariance distance')
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('I am a caption.')
                with doc.create(SubFigure(position='h!',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.diversity_scores, metric_name='diversity',
                                      value_name='population')
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('I am a caption.')
                plot.add_caption('I am a caption.')

    def add_model_plot(self, doc, width, *args, **kwargs):
        with doc.create(Subsection('Model Plot')):
            with doc.create(Figure(position='h!')) as plot:
                self.best_model.plot(plot_density=True)
                plot.add_plot(width=NoEscape(width), *args, **kwargs)
                plot.add_caption('I am a caption.')

    def add_results(self, doc, width, *args, **kwargs):
        with doc.create(Section('Results')):
            doc.append('Here are some plots summarizing the kernel search.')
            self.add_model_scores(doc, width, *args, **kwargs)
            self.add_kernel_structure_subsection(doc, width, *args, **kwargs)
            self.add_population_subsection(doc, width, *args, **kwargs)
            if self.experiment.n_dims == 1:
                # If training data is 1D, show a plot.
                self.add_model_plot(doc, width, *args, **kwargs)

    def add_timing_report(self, doc):
        with doc.create(Section('Timing Report')):
            doc.append('Here is a summary of the execution time of various parts of the algorithm.')
            with doc.create(Center()) as centered:
                with centered.create(Tabu("X[r] X[r] X[r]", to="4in")) as data_table:
                    header_row = ["Section", "Time Taken (s)", "Time Taken Percentage"]
                    data_table.add_row(header_row, mapper=[bold])
                    data_table.add_hline()
                    labels, x, x_pct = self.experiment.get_timing_report()
                    # sort by time
                    for label, sec, pct in sorted(zip(labels, x, x_pct), key=lambda v: v[1], reverse=True):
                        row = ('%s %0.2f %0.2f%%' % (label, sec, pct)).split(' ')
                        data_table.add_row(row)

    def add_model_summary(self, doc):
        with doc.create(Subsection('Best Model Summary')):
            doc.append('This table contains various scores of the best model.')
            doc.append("\n")
            doc.append(VerticalSpace("1pt"))
            doc.append(LineBreak())
            with doc.create(Center()) as centered:
                with centered.create(Tabu("|c|c|c|c|c|c|", to="4in")) as data_table:
                    header_row = ["NLL", "NLL (normalized)", "Mean NLPD", "BIC", "AIC", "PL2"]
                    data_table.add_row(header_row, mapper=[bold])
                    data_table.add_hline()

                    nll = -self.best_model.log_likelihood()
                    nll_norm = log_likelihood_normalized(self.best_model)
                    mean_nlpd = np.mean(-self.best_model.log_predictive_density(self.X_test, self.y_test))
                    aic = AIC(self.best_model)
                    bic = BIC(self.best_model)
                    pl2_score = pl2(self.best_model)

                    row = ('%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f' % (nll, nll_norm, mean_nlpd, bic,
                                                                    aic, pl2_score)).split(' ')
                    data_table.add_row(row)

    def add_comparison(self, doc):
        with doc.create(Subsection('Comparison to Other Models')):
            doc.append('This table contains the RMSE of the best model and others.')
            doc.append("\n")
            doc.append(VerticalSpace("1pt"))
            doc.append(LineBreak())
            with doc.create(Center()) as centered:
                with centered.create(Tabu("|c|c|c|", to="4in")) as data_table:
                    header_row = ["Best Model", "Linear Regression", "Support Vector Regression"]
                    data_table.add_row(header_row, mapper=[bold])
                    data_table.add_hline()

                    rmse_best_model = compute_gpy_model_rmse(self.best_model, self.X_test, self.y_test)
                    rmse_lr = rmse_lin_reg(self.X_train, self.y_train, self.X_test, self.y_test)
                    rmse_svm = rmse_svr(self.X_train, self.y_train, self.X_test, self.y_test)

                    row = ('%0.3f %0.3f %0.3f' % (rmse_best_model, rmse_lr, rmse_svm)).split(' ')
                    data_table.add_row(row)

    def add_performance(self, doc):
        with doc.create(Section('Predictive Performance')):
            doc.append('An evaluation of the performance of the best model discovered.')
            self.add_model_summary(doc)
            self.add_comparison(doc)

    def add_exp_params(self, doc):
        with doc.create(Section('Experimental Parameters')):
            doc.append('This section contains the parameters of the experiment')
            # TODO: fill out this section

    def create_result_file(self, fname, width, *args, **kwargs):
        geometry_options = {"right": "2cm", "left": "2cm"}
        doc = Document(fname, geometry_options=geometry_options)

        doc.preamble.append(Command('title', 'CKS Experiment Results'))
        doc.preamble.append(Command('author', 'Automatically Generated'))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))

        self.add_overview(doc)
        self.add_results(doc, width, *args, **kwargs)
        self.add_timing_report(doc)
        self.add_performance(doc)
        self.add_performance(doc)

        doc.generate_pdf(clean_tex=True)

    def summarize_experiment(self):
        # Create results folder if it doesn't exist
        results_path = os.path.join('..', self.results_dir_name)
        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        # Create file path
        now = datetime.datetime.now()
        now_fmt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = now_fmt_str
        file_path = os.path.join(results_path, file_name)

        self.create_result_file(file_path, r'1\textwidth', dpi=300)

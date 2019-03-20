import datetime
import os
from typing import List

import numpy as np
from GPy.core import GP
from GPy.kern import RBF
from GPy.models import GPRegression
from pylatex import Document, Section, Figure, NoEscape, SubFigure, Center, Tabu, MiniPage, LineBreak, VerticalSpace, \
    Subsection, Command, HorizontalSpace
from pylatex.utils import bold, italic
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.autoks.kernel import kernel_to_infix, AKSKernel
from src.autoks.model import AIC, BIC, pl2, log_likelihood_normalized
from src.evalg.plotting import plot_distribution, plot_best_so_far


def compute_skmodel_rmse(model,
                         x_train: np.ndarray,
                         y_train: np.ndarray,
                         x_test: np.ndarray,
                         y_test: np.ndarray) -> float:
    """RMSE of a scikit-learn model.

    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def compute_gpy_model_rmse(model: GP,
                           x_test: np.ndarray,
                           y_test: np.ndarray) -> float:
    """RMSE of a GPy model.

    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    mean, var = model.predict(x_test)
    y_pred = mean
    return np.sqrt(mean_squared_error(y_test, y_pred))


def rmse_rbf(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a GPy RBF kernel.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model = GPRegression(x_train, y_train, kernel=RBF(input_dim=x_train.shape[1]))
    model.optimize()
    return compute_gpy_model_rmse(model, x_test, y_test)


def rmse_svr(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a Support Vector Machine for regression.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(SVR(kernel='rbf'), x_train, y_train, x_test, y_test)


def rmse_lin_reg(x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray) -> float:
    """RMSE of a linear regression model.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(LinearRegression(), x_train, y_train, x_test, y_test)


def rmse_knn(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a k-nearest neighbors regressor.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(KNeighborsRegressor(), x_train, y_train, x_test, y_test)


class ExperimentReportGenerator:
    aks_kernels: List[AKSKernel]
    x_test: np.ndarray
    y_test: np.ndarray
    results_dir_name: str

    def __init__(self, experiment, aks_kernels, x_test, y_test, results_dir_name='results'):
        self.experiment = experiment
        self.aks_kernels = aks_kernels
        self.results_dir_name = results_dir_name

        self.x_train = self.experiment.x_train
        self.y_train = self.experiment.y_train
        self.x_test = x_test
        self.y_test = y_test

        scored_kernels = [kernel for kernel in self.aks_kernels if kernel.evaluated]
        sorted_aks_kernels = sorted(scored_kernels, key=lambda k: k.score, reverse=True)
        self.best_aks_kernel = sorted_aks_kernels[0]
        self.best_kernel = self.best_aks_kernel.kernel
        self.best_model = self.experiment.gp_model.__class__(self.x_train, self.y_train, kernel=self.best_kernel)

        self.width = r'1\textwidth'
        self.dpi = 300

    def add_overview(self,
                     doc: Document,
                     title: str = 'Overview') -> None:
        """Add overview section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Section(title)):
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

    def add_model_scores(self,
                         doc: Document,
                         width: str,
                         title: str = 'Model Score Evolution',
                         *args,
                         **kwargs) -> \
            None:
        """Add model scores sub-section to document.

        :param doc:
        :param width:
        :param title:
        :param args:
        :param kwargs:
        :return:
        """
        with doc.create(Subsection(title)):
            doc.append('A summary of the distribution of model scores.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_best_so_far(self.experiment.best_scores)
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('A plot of the maximum score over each iteration.')
                plot.append(HorizontalSpace("10pt"))
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.mean_scores, self.experiment.std_scores,
                                      self.experiment.best_scores)
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('A distribution of the maximum model score, the mean model score, and standard \
                    deviation of models scores per iteration.')
                plot.add_caption('These two figures show the model scores. The left shows a best-so-far curve and the \
                right one shows a distribution of scores.')

    def add_kernel_structure_subsection(self,
                                        doc: Document,
                                        width: str,
                                        title: str = 'Kernel Structure Evolution',
                                        *args,
                                        **kwargs) -> None:
        """Add kernel structure sub-section to document.

        :param doc:
        :param width:
        :param title:
        :param args:
        :param kwargs:
        :return:
        """
        with doc.create(Subsection(title)):
            doc.append('A summary of the structure of kernels searched.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_distribution(self.experiment.median_n_hyperparameters, self.experiment.std_n_hyperparameters,
                                      self.experiment.best_n_hyperparameters, value_name='median',
                                      metric_name='# Hyperparameters')
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('A plot of the number of hyperparameters for each iteration of the best model, \
                    the median number of hyperparameters, and the standard deviation.')
                plot.append(HorizontalSpace("10pt"))
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.median_n_operands,
                                      self.experiment.std_n_operands, self.experiment.best_n_operands,
                                      value_name='median', metric_name='# Operands')
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('A plot of the number of operands (number of 1-D kernels) over time including \
                    the best model, the median number of operands, and the standard deviation.')
                plot.add_caption('These two figures show how the structure of the compositional kernels changed over \
                time. The left figure shows the hyperparameter distribution and the right one shows operand \
                distribution.')

    def add_population_subsection(self,
                                  doc: Document,
                                  width: str,
                                  title: str = 'Population Evolution',
                                  *args,
                                  **kwargs) -> None:
        """Add population sub-section to document.

        :param doc:
        :param width:
        :param title:
        :param args:
        :param kwargs:
        :return:
        """
        with doc.create(Subsection(title)):
            doc.append('A summary of the population of kernels searched.')
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as left:
                    plot_distribution(self.experiment.mean_cov_dists, self.experiment.std_cov_dists,
                                      metric_name='covariance distance')
                    left.add_plot(width=NoEscape(width), *args, **kwargs)
                    left.add_caption('This plot shows the mean Euclidean covariance distance over time of all \
                    pairs of kernel matrices. It represents the heterogeneity of the population.')
                plot.append(HorizontalSpace("10pt"))
                with doc.create(SubFigure(position='t',
                                          width=NoEscape(r'0.45\linewidth'))) as right:
                    plot_distribution(self.experiment.diversity_scores, metric_name='diversity',
                                      value_name='population')
                    right.add_plot(width=NoEscape(width), *args, **kwargs)
                    right.add_caption('This plot shows the mean Euclidean distance of all pairs of kernel expressions \
                    in additive form. It represents the diversity/heterogeneity of the population.')
                plot.add_caption('Two figures showing the evolution of the population heterogeneity.')

    def add_model_plot(self,
                       doc: Document,
                       width: str,
                       title: str = 'Model Plot',
                       *args,
                       **kwargs) -> None:
        """Add model plot sub-section to document.

        :param doc:
        :param width:
        :param title:
        :param args:
        :param kwargs:
        :return:
        """
        with doc.create(Subsection(title)):
            with doc.create(Figure(position='h!')) as plot:
                self.best_model.plot(plot_density=True)
                plot.add_plot(width=NoEscape(width), *args, **kwargs)
                plot.add_caption('A plot of the fit of the best Gaussian Process discovered in the search.')

    def add_results(self,
                    doc: Document,
                    width: str,
                    title: str = 'Results',
                    *args,
                    **kwargs) -> None:
        """Add results sub-section to document.

        :param doc:
        :param width:
        :param title:
        :param args:
        :param kwargs:
        :return:
        """
        with doc.create(Section(title)):
            doc.append('Here are some plots summarizing the kernel search.')
            self.add_model_scores(doc, width, *args, **kwargs)
            self.add_kernel_structure_subsection(doc, width, *args, **kwargs)
            self.add_population_subsection(doc, width, *args, **kwargs)
            if self.experiment.n_dims == 1:
                # If training data is 1D, show a plot.
                self.add_model_plot(doc, width, *args, **kwargs)

    def add_timing_report(self,
                          doc: Document,
                          title: str = 'Timing Report') -> None:
        """Add timing report sub-section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Section(title)):
            doc.append('Here is a summary of the execution time of various parts of the algorithm.')
            with doc.create(Center()) as centered:
                with centered.create(Tabu("x[r] x[r] x[r]", to="4in")) as data_table:
                    header_row = ["Section", "Time Taken (s)", "Time Taken Percentage"]
                    data_table.add_row(header_row, mapper=[bold])
                    data_table.add_hline()
                    labels, x, x_pct = self.experiment.get_timing_report()
                    # sort by time
                    for label, sec, pct in sorted(zip(labels, x, x_pct), key=lambda v: v[1], reverse=True):
                        row = ('%s %0.2f %0.2f%%' % (label, sec, pct)).split(' ')
                        data_table.add_row(row)

    def add_model_summary(self,
                          doc: Document,
                          title: str = 'Best Model Summary') -> None:
        """Add model summary sub-section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Subsection(title)):
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
                    mean_nlpd = np.mean(-self.best_model.log_predictive_density(self.x_test, self.y_test))
                    aic = AIC(self.best_model)
                    bic = BIC(self.best_model)
                    pl2_score = pl2(self.best_model)

                    row = ('%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f' % (nll, nll_norm, mean_nlpd, bic,
                                                                    aic, pl2_score)).split(' ')
                    data_table.add_row(row)

    def add_comparison(self,
                       doc: Document,
                       title: str = 'Comparison to Other Models') -> None:
        """Add comparison sub-section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Subsection(title)):
            doc.append('This table contains the RMSE of the best model and others.')
            doc.append("\n")
            doc.append(VerticalSpace("1pt"))
            doc.append(LineBreak())
            with doc.create(Center()) as centered:
                with centered.create(Tabu("|c|c|c|c|c|", to="4in")) as data_table:
                    header_row = ["Best Model", "Linear Regression", "Support Vector Regression", "GP (RBF kernel)",
                                  "k-NN Regression"]
                    data_table.add_row(header_row, mapper=[bold])
                    data_table.add_hline()

                    rmse_best_model = compute_gpy_model_rmse(self.best_model, self.x_test, self.y_test)
                    rmse_lr = rmse_lin_reg(self.x_train, self.y_train, self.x_test, self.y_test)
                    rmse_svm = rmse_svr(self.x_train, self.y_train, self.x_test, self.y_test)
                    se_rmse = rmse_rbf(self.x_train, self.y_train, self.x_test, self.y_test)
                    knn_rmse = rmse_knn(self.x_train, self.y_train, self.x_test, self.y_test)

                    row = ('%0.3f %0.3f %0.3f %0.3f %0.3f' %
                           (rmse_best_model, rmse_lr, rmse_svm, se_rmse, knn_rmse)).split(' ')
                    data_table.add_row(row)

    def add_performance(self,
                        doc: Document,
                        title: str = 'Predictive Performance') -> None:
        """Add performance section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Section(title)):
            doc.append('An evaluation of the performance of the best model discovered.')
            self.add_model_summary(doc)
            self.add_comparison(doc)

    def add_exp_params(self,
                       doc: Document,
                       title: str = 'Experimental Parameters') -> None:
        """Add experimental parameter section to document.

        :param doc:
        :param title:
        :return:
        """
        with doc.create(Section(title)):
            doc.append('This section contains the parameters of the experiment')
            # TODO: fill out this section

    def create_result_file(self,
                           fname: str,
                           width: str,
                           title: str,
                           author: str,
                           *args,
                           **kwargs) -> None:
        """Create a experiment result file.

        :param fname:
        :param width:
        :param title:
        :param author:
        :param args:
        :param kwargs:
        :return:
        """
        geometry_options = {"right": "2cm", "left": "2cm"}
        doc = Document(fname, geometry_options=geometry_options)

        doc.preamble.append(Command('title', title))
        doc.preamble.append(Command('author', author))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))

        self.add_overview(doc)
        self.add_results(doc, width, *args, **kwargs)
        self.add_timing_report(doc)
        self.add_performance(doc)
        self.add_exp_params(doc)

        doc.generate_pdf(clean_tex=True)

    def summarize_experiment(self,
                             title: str = 'Experiment Results',
                             author: str = 'Automatically Generated') -> None:
        """Summarize experiment by creating the result file.

        :param title:
        :param author:
        :return:
        """
        # Create results folder if it doesn't exist
        results_path = os.path.join('..', self.results_dir_name)
        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        # Create file path
        now = datetime.datetime.now()
        now_fmt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = now_fmt_str
        file_path = os.path.join(results_path, file_name)

        self.create_result_file(file_path, self.width, title, author, dpi=self.dpi)

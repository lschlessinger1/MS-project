import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from autoks import model
from autoks.experiment import Experiment
from autoks.grammar import CKSGrammar

# Set random seed for reproducibility.
np.random.seed(4096)

X, y, gt = make_regression(n_features=1, n_informative=1, n_samples=100, coef=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if X.shape[1] > 1:
    base_kernels = ['SE', 'RQ']
else:
    base_kernels = ['SE', 'RQ', 'LIN', 'PER']
grammar = CKSGrammar(n_parents=1)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -model.BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent for CKS
optimizer = 'scg'

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train.reshape(-1, 1), eval_budget=50, debug=True,
                        verbose=True, optimizer=optimizer)
aks_kernels = experiment.kernel_search()

# View results of experiment
experiment.plot_best_scores()
experiment.plot_score_summary()
experiment.plot_n_hyperparams_summary()
experiment.plot_n_operands_summary()
experiment.plot_cov_dist_summary()
experiment.plot_kernel_diversity_summary()
experiment.timing_report()

sorted_aks_kernels = sorted(aks_kernels, key=lambda x: x.score, reverse=True)
best_aks_kernel = sorted_aks_kernels[0]
best_kernel = best_aks_kernel.kernel
print('Best kernel:')
best_aks_kernel.pretty_print()
print('In full form:')
best_aks_kernel.print_full()

best_model = experiment.gp_model.__class__(experiment.X, experiment.y, kernel=best_kernel)

# If training data is 1D, show a plot.
if best_model.input_dim == 1:
    best_model.plot(plot_density=True)
    plt.show()

mean, var = best_model.predict(X_test)
y_pred = mean
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE = %.3f' % rmse)

mean_lpd = np.mean(best_model.log_predictive_density(X_test, y_test))
print('Negative log predictive density = %.3f' % -mean_lpd)

# Compare against linear and support vector regression.
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
rmse_lin_reg = np.sqrt(mean_squared_error(y_test, y_pred_lin_reg))

svr = SVR(kernel='rbf').fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

print('RMSE Linear Regression = %.3f' % rmse_lin_reg)
print('RMSE SVM = %.3f' % rmse_svr)

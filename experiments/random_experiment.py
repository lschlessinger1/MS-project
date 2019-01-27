import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from experiments.util import synthetic_data

top_path = os.path.abspath('..')
if top_path not in sys.path:
    print('Adding to sys.path %s' % top_path)
    sys.path.append(top_path)

from autoks import model
from autoks.Experiment import Experiment
from autoks.grammar import RandomGrammar

# Set random seed for reproducibility.
np.random.seed(4096)

X, y = synthetic_data.generate_data(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if X.shape[1] > 1:
    base_kernels = ['SE', 'RQ']
else:
    base_kernels = ['SE', 'RQ', 'LIN', 'PER']
grammar = RandomGrammar(n_parents=4)


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

sorted_aks_kernels = sorted(aks_kernels, key=lambda x: x.score, reverse=True)
best_aks_kernel = sorted_aks_kernels[0]
best_kernel = best_aks_kernel.kernel
print('Best kernel:')
best_aks_kernel.pretty_print()

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

svr = SVR().fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

print('RMSE Linear Regression = %.3f' % rmse_lin_reg)
print('RMSE SVM = %.3f' % rmse_svr)

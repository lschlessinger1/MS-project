import numpy as np
from sklearn.model_selection import train_test_split

from autoks import model
from autoks.experiment import Experiment
from autoks.grammar import RandomGrammar
from experiments.util import synthetic_data

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

# use conjugate gradient descent
optimizer = 'scg'

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, eval_budget=50, debug=True,
                        verbose=True, optimizer=optimizer)
experiment.run()

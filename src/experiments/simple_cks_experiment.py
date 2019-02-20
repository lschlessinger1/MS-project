import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

top_path = os.path.abspath('..')
if top_path not in sys.path:
    print('Adding to sys.path %s' % top_path)
    sys.path.append(top_path)

from src.autoks import model
from src.autoks.experiment import Experiment
from src.autoks.grammar import CKSGrammar
from src.experiments.util import synthetic_data

# Set random seed for reproducibility.
np.random.seed(4096)

# Create synthetic dataset
X, y = synthetic_data.sinosoid_2(n_samples=100, n_dims=1)
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

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, eval_budget=50, debug=True,
                        verbose=True, optimizer=optimizer)
experiment.run()

import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import RandomGrammar
from src.autoks.model import BIC
from src.experiments.util.synthetic_data import RegressionGenerator

# Set random seed for reproducibility.
np.random.seed(4096)

generator = RegressionGenerator()
X, y = generator.gen_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if X.shape[1] > 1:
    base_kernels = ['SE', 'RQ']
else:
    base_kernels = ['SE', 'RQ', 'LIN', 'PER']
grammar = RandomGrammar(n_parents=4, max_candidates=0, max_offspring=1000)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent
optimizer = 'scg'

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, eval_budget=50, debug=True,
                        verbose=True, optimizer=optimizer)
experiment.run(title='Random Experiment')

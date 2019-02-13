import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

top_path = os.path.abspath('..')
if top_path not in sys.path:
    print('Adding to sys.path %s' % top_path)
    sys.path.append(top_path)

from autoks.experiment import Experiment
from autoks.grammar import EvolutionaryGrammar
from autoks.model import log_likelihood_normalized
from experiments.util import synthetic_data

# Set random seed for reproducibility.
np.random.seed(4096)

X, y = synthetic_data.generate_data(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if X.shape[1] > 1:
    base_kernels = ['SE', 'RQ']
else:
    base_kernels = ['SE', 'RQ', 'LIN', 'PER']
grammar = EvolutionaryGrammar(n_parents=4)

objective = log_likelihood_normalized

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_train, y_train, eval_budget=50,
                        debug=True, verbose=True)
experiment.run()

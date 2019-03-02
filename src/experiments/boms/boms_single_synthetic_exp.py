import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import BOMSGrammar
from src.autoks.model import log_likelihood_normalized
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
grammar = BOMSGrammar()

objective = log_likelihood_normalized

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_train, y_train, eval_budget=50,
                        debug=True, verbose=True)
experiment.run(title='Simple BOMS Experiment')

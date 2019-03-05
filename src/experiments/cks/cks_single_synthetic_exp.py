import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks import model
from src.autoks.experiment import Experiment
from src.autoks.grammar import CKSGrammar
from src.experiments.util.synthetic_data import Sinosoid2Generator

# Set random seed for reproducibility.
np.random.seed(4096)

# Create synthetic dataset
generator = Sinosoid2Generator(n_samples=100, input_dim=1)
x, y = generator.gen_dataset()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])
grammar = CKSGrammar(n_parents=1, max_candidates=0, max_offspring=1000)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -model.BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent for CKS
optimizer = 'scg'

experiment = Experiment(grammar, objective, base_kernels, x_train, y_train, x_test, y_test, eval_budget=50, debug=True,
                        verbose=True, optimizer=optimizer)
experiment.run(title='Simple CKS Experiment')

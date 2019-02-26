import numpy as np

from src.autoks import model
from src.autoks.grammar import CKSGrammar
from src.experiments.util.data_util import KnownGPGenerator, cks_known_kernels, run_experiments

# Set random seed for reproducibility.
np.random.seed(4096)

grammar = CKSGrammar(n_parents=1)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -model.BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent for CKS
optimizer = 'scg'

# Create synthetic dataset generators
noise_vars = [10 ** i for i in range(-1, 2)]
generators = [KnownGPGenerator(kernel, var, 100) for var in noise_vars for kernel in cks_known_kernels()]

run_experiments(generators, grammar, objective, base_kernels=None, eval_budget=50, debug=True, verbose=True,
                optimizer=optimizer)
import numpy as np

from src.autoks.grammar import RandomGrammar
from src.autoks.model import BIC
from src.experiments.util.data_util import gen_dataset_paths, FileDatasetGenerator, run_experiments

# Set random seed for reproducibility.
np.random.seed(4096)

grammar = RandomGrammar(n_parents=4)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent
optimizer = 'scg'

data_paths = gen_dataset_paths(data_dir='../data')
generators = [FileDatasetGenerator(path) for path in data_paths]

run_experiments(generators, grammar, objective, base_kernels=None, eval_budget=50, debug=True, verbose=True,
                optimizer=optimizer)

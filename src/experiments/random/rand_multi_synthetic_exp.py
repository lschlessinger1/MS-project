import numpy as np

from src.autoks.grammar import RandomGrammar
from src.autoks.model import BIC
from src.experiments.util.data_util import run_experiments
from src.experiments.util.synthetic_data import Sinosoid1Dataset, Sinosoid2Dataset, SimplePeriodic1dDataset

# Set random seed for reproducibility.
np.random.seed(4096)

grammar = RandomGrammar(n_parents=4, max_candidates=0, max_offspring=1000)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

# use conjugate gradient descent
optimizer = 'scg'

generators = [Sinosoid1Dataset(), Sinosoid2Dataset(), SimplePeriodic1dDataset()]

run_experiments(generators, grammar, objective, base_kernels=None, eval_budget=50, debug=True, verbose=True,
                optimizer=optimizer)

import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.backend.model import BIC, log_likelihood_normalized
from src.autoks.core.grammar import CKSGrammar
from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.autoks.core.model_selection.cks_model_selector import CKSModelSelector
from src.autoks.tracking import ModelSearchTracker
# Set random seed for reproducibility.
from src.datasets.synthetic.synthetic_data import CubicSine1dDataset

np.random.seed(4096)

dataset = CubicSine1dDataset(n_samples=60)
dataset.load_or_generate_data()
x, y = dataset.x, dataset.y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

n_dims = x.shape[1]
grammar = CKSGrammar(n_dims)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the fitness_fn.
fitness_fn = log_likelihood_normalized

tracker = ModelSearchTracker(grammar.base_kernel_names)

model_selector = CKSModelSelector(grammar, fitness_fn, additive_form=False)

experiment = ModelSearchExperiment(x_train, y_train, model_selector, x_test, y_test, tracker)
experiment.run(eval_budget=8)

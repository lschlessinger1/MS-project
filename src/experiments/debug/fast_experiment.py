import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.backend.model import BIC
from src.autoks.core.grammar import RandomGrammar
from src.autoks.core.kernel_selection import CKS_kernel_selector
from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.autoks.core.model_selection import RandomModelSelector
from src.autoks.tracking import ModelSearchTracker
from src.experiments.util.synthetic_data import CubicSine1dDataset

# Set random seed for reproducibility.
np.random.seed(4096)

dataset = CubicSine1dDataset(n_samples=60)
x, y = dataset.load_or_generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

n_dims = x.shape[1]
grammar = RandomGrammar(n_dims)
kernel_selector = CKS_kernel_selector(n_parents=1)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

tracker = ModelSearchTracker(grammar.base_kernel_names)

model_selector = RandomModelSelector(grammar, kernel_selector, objective, eval_budget=8,
                                     tabu_search=False, debug=True, verbose=True, additive_form=False,
                                     active_set_callback=tracker.active_set_callback,
                                     eval_callback=tracker.evaluations_callback,
                                     expansion_callback=tracker.expansion_callback)

experiment = ModelSearchExperiment(x_train, y_train, model_selector, x_test, y_test, tracker)
experiment.run()

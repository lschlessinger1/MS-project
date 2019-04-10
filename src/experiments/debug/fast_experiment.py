import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import RandomGrammar, CKSGrammar
from src.autoks.kernel_selection import CKS_kernel_selector
from src.autoks.model import BIC
from src.experiments.util.synthetic_data import CubicSine1dDataset

# Set random seed for reproducibility.
np.random.seed(4096)

dataset = CubicSine1dDataset(n_samples=100)
x, y = dataset.load_or_generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])
grammar = RandomGrammar()
kernel_selector = CKS_kernel_selector(n_parents=1)


def negative_BIC(m):
    """Computes the negative of the Bayesian Information Criterion (BIC)."""
    return -BIC(m)


# Use the negative BIC because we want to maximize the objective.
objective = negative_BIC

experiment = Experiment(grammar, kernel_selector, objective, base_kernels, x_train, y_train, x_test, y_test,
                        eval_budget=8, debug=True, verbose=True, additive_form=False)
experiment.run(title='Fast Experiment', create_report=False)

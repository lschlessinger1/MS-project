import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import BOMSGrammar, CKSGrammar
from src.autoks.hyperprior import boms_hyperpriors
from src.autoks.kernel_selection import BOMS_kernel_selector
from src.autoks.model import log_likelihood_normalized
from src.autoks.query_strategy import BOMSInitQueryStrategy
from src.experiments.util.synthetic_data import SyntheticRegressionDataset

# Set random seed for reproducibility.
np.random.seed(4096)

dataset = SyntheticRegressionDataset(input_dim=2)
x, y = dataset.load_or_generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])
grammar = BOMSGrammar()
kernel_selector = BOMS_kernel_selector()
hyperpriors = boms_hyperpriors()

objective = log_likelihood_normalized

init_qs = BOMSInitQueryStrategy()

experiment = Experiment(grammar, kernel_selector, objective, base_kernels, x_train, y_train, x_train, y_train,
                        eval_budget=50, debug=True, verbose=True, hyperpriors=hyperpriors, init_query_strat=init_qs)
experiment.run(title='Simple BOMS Experiment')

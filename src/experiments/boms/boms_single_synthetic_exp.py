import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import BOMSGrammar, CKSGrammar
from src.autoks.model import log_likelihood_normalized
from src.experiments.util.synthetic_data import RegressionGenerator

# Set random seed for reproducibility.
np.random.seed(4096)

generator = RegressionGenerator()
x, y = generator.gen_dataset()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])
grammar = BOMSGrammar()

objective = log_likelihood_normalized

experiment = Experiment(grammar, objective, base_kernels, x_train, y_train, x_train, y_train, eval_budget=50,
                        debug=True, verbose=True)
experiment.run(title='Simple BOMS Experiment')

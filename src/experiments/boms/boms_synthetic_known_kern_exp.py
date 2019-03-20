import numpy as np

from src.autoks.grammar import BOMSGrammar
from src.autoks.hyperprior import boms_hyperpriors
from src.autoks.kernel_selection import BOMS_kernel_selector
from src.autoks.model import log_likelihood_normalized
from src.experiments.util.data_util import KnownGPDataset, cks_known_kernels, run_experiments

# Set random seed for reproducibility.
np.random.seed(4096)

grammar = BOMSGrammar()
kernel_selector = BOMS_kernel_selector()
objective = log_likelihood_normalized
hyperpriors = boms_hyperpriors()

# Create synthetic dataset generators
noise_vars = [10 ** i for i in range(-1, 2)]
datasets = [KnownGPDataset(kernel, var, 100) for var in noise_vars for kernel in cks_known_kernels()]

run_experiments(datasets, grammar, kernel_selector, objective, base_kernels=None, eval_budget=50,
                debug=True, verbose=True, hyperpriors=hyperpriors)

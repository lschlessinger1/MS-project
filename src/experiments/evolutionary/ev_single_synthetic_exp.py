import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import EvolutionaryGrammar, CKSGrammar
from src.autoks.kernel import get_all_1d_kernels
from src.autoks.kernel_selection import evolutionary_kernel_selector
from src.autoks.model import log_likelihood_normalized
from src.evalg.genprog import SubtreeExchangeBinaryRecombinator, GrowMutator
from src.evalg.vary import CrossMutPopOperator, CrossoverVariator, MutationVariator
from src.experiments.util.synthetic_data import SyntheticRegressionDataset

# Set random seed for reproducibility.
np.random.seed(4096)

generator = SyntheticRegressionDataset(n_samples=100)
x, y = generator.load_or_generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])

mutator = GrowMutator(operands=get_all_1d_kernels(base_kernels, x.shape[1]))
recombinator = SubtreeExchangeBinaryRecombinator()
cx_variator = CrossoverVariator(recombinator, n_offspring=10)
mut_variator = MutationVariator(mutator)
variators = [cx_variator, mut_variator]
pop_operator = CrossMutPopOperator(variators)
grammar = EvolutionaryGrammar(population_operator=pop_operator)

kernel_selector = evolutionary_kernel_selector(n_parents=4, max_offspring=1000)

objective = log_likelihood_normalized

experiment = Experiment(grammar, kernel_selector, objective, base_kernels, x_train, y_train, x_train, y_train,
                        eval_budget=50, additive_form=True, debug=True, verbose=True)
experiment.run(title='Simple Evolutionary Experiment')

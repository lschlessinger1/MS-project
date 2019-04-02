import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import EvolutionaryGrammar, CKSGrammar
from src.autoks.kernel import get_all_1d_kernels
from src.autoks.kernel_selection import evolutionary_kernel_selector
from src.autoks.model import log_likelihood_normalized
from src.evalg.genprog import HalfAndHalfMutator, OnePointRecombinator, HalfAndHalfGenerator
from src.evalg.vary import CrossMutPopOperator, CrossoverVariator, MutationVariator
from src.experiments.util.synthetic_data import Sinosoid2Dataset

# Set random seed for reproducibility.
np.random.seed(4096)

dataset = Sinosoid2Dataset(n_samples=100, input_dim=1)
x, y = dataset.load_or_generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(x.shape[1])

n_offspring = 10
pop_size = 25

mutator = HalfAndHalfMutator(operands=get_all_1d_kernels(base_kernels, x.shape[1]))
recombinator = OnePointRecombinator()
cx_variator = CrossoverVariator(recombinator, n_offspring=n_offspring)
mut_variator = MutationVariator(mutator)
variators = [cx_variator, mut_variator]
pop_operator = CrossMutPopOperator(variators)

grammar = EvolutionaryGrammar(population_operator=pop_operator)
initializer = HalfAndHalfGenerator(binary_operators=grammar.operators, max_depth=1, operands=mutator.operands)
grammar.initializer = initializer
grammar.n_init_trees = pop_size

kernel_selector = evolutionary_kernel_selector(n_parents=4, max_offspring=pop_size)

objective = log_likelihood_normalized

experiment = Experiment(grammar, kernel_selector, objective, base_kernels, x_train, y_train, x_train, y_train,
                        eval_budget=50, additive_form=False, debug=True, verbose=True, max_null_queries=50,
                        max_same_expansions=50, tabu_search=False)
experiment.run(title='Simple Evolutionary Experiment')

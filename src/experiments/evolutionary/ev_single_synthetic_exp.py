import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import EvolutionaryGrammar, CKSGrammar
from src.autoks.kernel import get_all_1d_kernels
from src.autoks.model import log_likelihood_normalized
from src.evalg.genprog import SubtreeExchangeBinaryRecombinator, GrowMutator
from src.evalg.selection import TruncationSelector, AllSelector
from src.evalg.vary import CrossMutPopOperator, CrossoverVariator, MutationVariator
from src.experiments.util.synthetic_data import RegressionGenerator

# Set random seed for reproducibility.
np.random.seed(4096)

generator = RegressionGenerator(n_samples=100)
X, y = generator.gen_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

base_kernels = CKSGrammar.get_base_kernels(X.shape[1])

parent_selector = TruncationSelector(10)
offspring_selector = AllSelector(1)

mutator = GrowMutator(operands=get_all_1d_kernels(base_kernels, X.shape[1]))
recombinator = SubtreeExchangeBinaryRecombinator()
cx_variator = CrossoverVariator(recombinator, n_offspring=10)
mut_variator = MutationVariator(mutator)
variators = [cx_variator, mut_variator]
pop_operator = CrossMutPopOperator(variators)
grammar = EvolutionaryGrammar(n_parents=4, parent_selector=parent_selector, offspring_selector=offspring_selector,
                              population_operator=pop_operator, max_candidates=0, max_offspring=1000)

objective = log_likelihood_normalized

experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_train, y_train, eval_budget=50,
                        additive_form=True, debug=True, verbose=True)
experiment.run(title='Simple Evolutionary Experiment')

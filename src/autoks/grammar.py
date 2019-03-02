from typing import List

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.kernel import get_all_1d_kernels, create_1d_kernel, AKSKernel, remove_duplicate_kernels, tree_to_kernel
from src.evalg.selection import TruncationSelector, Selector, AllSelector
from src.evalg.vary import PopulationOperator


class BaseGrammar:
    DEFAULT_OPERATORS = ['+', '*']

    def __init__(self, n_parents: int, max_candidates: int, max_offspring: int):
        self.n_parents = n_parents  # number of parents to expand each round
        self.max_candidates = max_candidates  # Max. number of un-evaluated models to keep each round
        self.max_offspring = max_offspring  # Max. number of models to keep each round
        self.operators = BaseGrammar.DEFAULT_OPERATORS

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int) -> List[AKSKernel]:
        """Initialize kernels.

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self, kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False) -> \
            List[AKSKernel]:
        """Get next round of candidate kernels from current kernels.

        :param kernels:
        :param kernel_families:
        :param n_dims: number of dimensions
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def select_parents(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Select parent kernels (default is top k kernels by objective).

        :param kernels:
        :return:
        """
        selector = TruncationSelector(self.n_parents)
        return selector.select(kernels, np.array([k.score for k in kernels]))

    def select_offspring(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """Select next round of kernels (default is select all).

        :param kernels:
        :return:
        """
        selector = AllSelector(self.max_offspring)
        return selector.select(kernels)

    def prune_candidates(self, kernels: List[AKSKernel], acq_scores) -> List[AKSKernel]:
        """Remove candidates from kernel list (by default remove none).

        :param kernels:
        :param acq_scores:
        :return:
        """
        # by default, we have no pruning
        selector = AllSelector(self.max_candidates)
        return selector.select(kernels)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class EvolutionaryGrammar(BaseGrammar):

    def __init__(self, n_parents: int, max_candidates: int, max_offspring: int, population_operator: PopulationOperator,
                 parent_selector: Selector, offspring_selector: Selector):
        super().__init__(n_parents, max_candidates, max_offspring)
        self.population_operator = population_operator
        self.parent_selector = parent_selector
        self.offspring_selector = offspring_selector

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int) -> List[AKSKernel]:
        """Naive initialization of all SE_i and RQ_i (for every dimension).

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False) -> \
            List[AKSKernel]:
        """Perform crossover and mutation.

        :param aks_kernels: list of AKSKernels
        :param kernel_families: base kernels
        :param n_dims:
        :param verbose:
        :return:
        """
        if verbose:
            print('Seed kernels:')
            for k in aks_kernels:
                k.pretty_print()

        # Convert GPy kernels to BinaryTrees
        trees = [aks_kernel.to_binary_tree() for aks_kernel in aks_kernels]

        # Mutate/Crossover Trees
        offspring = self.population_operator.create_offspring(trees)

        # Convert Trees back to GPy kernels, then to AKSKernels
        kernels = [tree_to_kernel(tree) for tree in offspring]

        new_kernels = remove_duplicate_kernels(kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print(f'{len(new_kernels)} Newly expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def select_parents(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """See parent docstring.

        :param kernels:
        :return:
        """
        return self.parent_selector.select(kernels, [k.score for k in kernels])

    def select_offspring(self, kernels: List[AKSKernel]) -> List[AKSKernel]:
        """See parent docstring.

        :param kernels:
        :return:
        """
        return self.offspring_selector.select(kernels, [k.score for k in kernels])

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class BOMSGrammar(BaseGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """

    def __init__(self, n_parents: int = 1, max_candidates: int = 600, max_offspring: int = 1000):
        super().__init__(n_parents, max_candidates, max_offspring)

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int) -> List[AKSKernel]:
        """Initialize kernels according to number of dimensions.

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        # {SE, RQ, LIN, PER} if dataset is 1D
        # {SE_i} + {RQ_i} otherwise
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False) -> \
            List[AKSKernel]:
        """Greedy and exploratory expansion of kernels.

        :param aks_kernels: list of AKSKernels
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :return:
        """
        if verbose:
            print('Seed kernels:')
            for k in aks_kernels:
                k.pretty_print()

        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        rw_kerns = BOMSGrammar.random_walk_kernels(n_dims, kernel_families)

        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        scored_kernels = [kernel for kernel in aks_kernels if kernel.scored]
        best_kern = sorted(scored_kernels, key=lambda x: x.score, reverse=True)[0]
        greedy_kerns = BOMSGrammar.greedy_kernels(best_kern, n_dims, kernel_families)

        new_kernels = rw_kerns + greedy_kerns
        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print(f'{len(new_kernels)} Newly expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def prune_candidates(self, active_set: List[AKSKernel], acq_scores) -> List[AKSKernel]:
        """Select best kernels according to expected improvement.

        :param active_set:
        :param acq_scores:
        :return:
        """
        selector = TruncationSelector(self.max_candidates)
        # prioritize keeping scored models
        augmented_scores = [k.score if k.scored and not k.nan_scored else -np.inf for k in active_set]
        return selector.select(active_set, np.array(augmented_scores))

    @staticmethod
    def random_walk_kernels(n_dims: int, base_kernels: List[str], t_prob: float = 1 / 3., n_walks: int = 15) -> \
            List[AKSKernel]:
        """Geometric random walk kernels.

        :param n_dims:
        :param base_kernels:
        :param t_prob: termination probability
        :param n_walks: number of random walks
        :return:
        """
        # geometric random walk
        n_steps = np.random.geometric(p=t_prob, size=n_walks)
        rw_kernels = []
        for n in n_steps:
            # first expansion of empty kernel is all 1d kernels
            kernels = get_all_1d_kernels(base_kernels, n_dims)
            random_kernel = np.random.choice(kernels)
            for i in range(1, n):
                kernels = CKSGrammar.expand_full_kernel(random_kernel, n_dims, base_kernels,
                                                        BaseGrammar.DEFAULT_OPERATORS)
                random_kernel = np.random.choice(kernels)
                rw_kernels.append(random_kernel)

        return rw_kernels

    @staticmethod
    def greedy_kernels(best_kernel: AKSKernel, n_dims, base_kernels) -> List[AKSKernel]:
        """Single expansion of CKS Grammar.

        :param best_kernel:
        :param n_dims:
        :param base_kernels:
        :return:
        """
        new_kernels = CKSGrammar.expand_full_kernel(best_kernel.kernel, n_dims, base_kernels,
                                                    CKSGrammar.DEFAULT_OPERATORS)

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self, n_parents: int, max_candidates: int, max_offspring: int):
        super().__init__(n_parents, max_candidates, max_offspring)

    @staticmethod
    def get_base_kernels(n_dims):
        if n_dims > 1:
            return ['SE', 'RQ']
        else:
            return ['SE', 'RQ', 'LIN', 'PER']

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int) -> List[AKSKernel]:
        """Initialize with all base kernel families applied to all input dimensions.

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False) -> \
            List[AKSKernel]:
        """Greedy expansion of nodes.

        :param aks_kernels:
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :return:
        """
        # choose highest scoring kernel (using BIC) and expand it by applying all possible operators
        # CFG:
        # 1) Any subexpression S can be replaced with S + B, where B is any base kernel family.
        # 2) Any subexpression S can be replaced with S x B, where B is any base kernel family.
        # 3) Any base kernel B may be replaced with any other base kernel family B'
        if verbose:
            print('Seed kernels:')
            for k in aks_kernels:
                k.pretty_print()

        new_kernels = []
        for aks_kernel in aks_kernels:
            kernel = aks_kernel.kernel
            kernels_expanded = CKSGrammar.expand_full_kernel(kernel, n_dims, kernel_families, self.operators)
            new_kernels += kernels_expanded

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print(f'{len(new_kernels)} Newly expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    @staticmethod
    def expand_single_kernel(kernel: Kern, n_dims: int, base_kernels: List[str], operators: List[str]) -> \
            List[Kern]:
        """Expand a single kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param operators:
        :return:
        """
        is_kernel = isinstance(kernel, Kern)
        if not is_kernel:
            raise TypeError(f'Unknown kernel type {kernel.__class__.__name__}')

        kernels = []

        is_combo_kernel = isinstance(kernel, CombinationKernel)
        is_base_kernel = is_kernel and not is_combo_kernel

        for op in operators:
            for d in range(n_dims):
                for base_kernel_name in base_kernels:
                    if op == '+':
                        kernels.append(kernel + create_1d_kernel(base_kernel_name, d))
                    elif op == '*':
                        kernels.append(kernel * create_1d_kernel(base_kernel_name, d))
                    else:
                        raise ValueError(f'Unknown operation {op}')
        for base_kernel_name in base_kernels:
            if is_base_kernel:
                kernels.append(create_1d_kernel(base_kernel_name, kernel.active_dims[0]))
        return kernels

    @staticmethod
    def expand_full_kernel(kernel: Kern, n_dims: int, base_kernels: List[str], operators: List[str]) -> List[Kern]:
        """Expand full kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param operators:
        :return:
        """
        result = CKSGrammar.expand_single_kernel(kernel, n_dims, base_kernels, operators)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in CKSGrammar.expand_full_kernel(operand, n_dims, base_kernels, operators):
                    new_operands = kernel.parts[:i] + [e] + kernel.parts[i + 1:]
                    new_operands = [op.copy() for op in new_operands]
                    if isinstance(kernel, Prod):
                        result.append(Prod(new_operands))
                    elif isinstance(kernel, Add):
                        result.append(Add(new_operands))
                    else:
                        raise TypeError(f'Unknown combination kernel class {kernel.__class__.__name__}')
        elif isinstance(kernel, Kern):
            # base kernel
            pass
        else:
            raise TypeError(f'Unknown kernel class {kernel.__class__.__name__}')
        return result

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class RandomGrammar(BaseGrammar):
    """Random grammar randomly expands nodes

    """

    def __init__(self, n_parents: int, max_candidates: int, max_offspring: int):

        super().__init__(n_parents, max_candidates, max_offspring)

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int) -> List[AKSKernel]:
        """Same initialization as CKS and BOMS

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False) -> \
            List[AKSKernel]:
        """Random expansion of nodes.

        :param aks_kernels:
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :return:
        """
        if verbose:
            print('Seed kernels:')
            for k in aks_kernels:
                k.pretty_print()

        new_kernels = []

        # for each kernel, randomly add or multiply a random 1D kernel
        for aks_kernel in aks_kernels:
            k = aks_kernel.kernel
            all_1d_kernels = get_all_1d_kernels(kernel_families, n_dims)

            random_op = np.random.choice(self.operators)
            random_1d_kernel = np.random.choice(all_1d_kernels)

            if random_op == '+':
                k += random_1d_kernel
            elif random_op == '*':
                k *= random_1d_kernel

            new_kernels.append(k)

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print(f'{len(new_kernels)} Newly expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'

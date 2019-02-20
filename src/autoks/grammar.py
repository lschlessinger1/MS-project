from typing import List

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.kernel import get_all_1d_kernels, create_1d_kernel, AKSKernel, remove_duplicate_kernels, tree_to_kernel
from src.evalg.selection import TruncationSelector, Selector, AllSelector
from src.evalg.vary import PopulationOperator


class BaseGrammar:

    def __init__(self, n_parents: int):
        self.n_parents = n_parents
        self.operators = ['+', '*']

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int):
        """ Initialize kernels

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self, kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False):
        """ Get next round of candidate kernels from current kernels

        :param kernels:
        :param kernel_families:
        :param n_dims: number of dimensions
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def select_parents(self, kernels: List[AKSKernel]):
        """ Select next round of models (default is top k kernels by objective)

        :param kernels:
        :return:
        """
        selector = TruncationSelector(self.n_parents)
        return selector.select(kernels, [k.score for k in kernels])

    def select_offspring(self, kernels: List[AKSKernel]):
        """ Select next round of models (default is select all)

        :param kernels:
        :return:
        """
        selector = AllSelector(self.n_parents)
        return selector.select(kernels)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class EvolutionaryGrammar(BaseGrammar):

    def __init__(self, n_parents: int, population_operator: PopulationOperator, parent_selector: Selector,
                 offspring_selector: Selector):
        super().__init__(n_parents)
        self.population_operator = population_operator
        self.parent_selector = parent_selector
        self.offspring_selector = offspring_selector

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int):
        """Naive initialization of all SE_i and RQ_i (for every dimension)

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False):
        """ Perform crossover and mutation

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
        new_kernels = [AKSKernel(kernel) for kernel in kernels]

        if verbose:
            print('Expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def select_parents(self, kernels: List[AKSKernel]):
        return self.parent_selector.select(kernels, [k.score for k in kernels])

    def select_offspring(self, kernels: List[AKSKernel]):
        return self.offspring_selector.select(kernels, [k.score for k in kernels])

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class BOMSGrammar(BaseGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """

    def __init__(self, n_parents: int = 600):
        super().__init__(n_parents)

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int):
        """ Initialize kernels according to number of dimensions

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

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False):
        """ Greedy and exploratory expansion of kernels

        :param aks_kernels: list of AKSKernels
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :return:
        """
        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        if verbose:
            print('Seed kernels:')
            for k in aks_kernels:
                k.pretty_print()

        # TODO: perform exploitation and exploration for all kernels
        # for now, just return a copy of the original kernels
        new_kernels = aks_kernels.copy()

        if verbose:
            print('Expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def select_offspring(self, active_set: List[AKSKernel]):
        """ Select top `n_parents` kernels according to expected improvement

        :param active_set:
        :return:
        """
        selector = TruncationSelector(self.n_parents)
        return selector.select(active_set, [k.score for k in active_set])

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self, n_parents: int):
        super().__init__(n_parents)

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int):
        """ Initialize with all base kernel families applied to all input dimensions

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False):
        """ Greedy expansion of nodes

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
            kernels_expanded = self.expand_full_kernel(kernel, n_dims, kernel_families)
            new_kernels += kernels_expanded

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print('Expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def expand_single_kernel(self, kernel: Kern, n_dims: int, base_kernels: List[str]):
        is_kernel = isinstance(kernel, Kern)
        if not is_kernel:
            raise TypeError(f'Unknown kernel type {kernel.__class__.__name__}')

        kernels = []

        is_combo_kernel = isinstance(kernel, CombinationKernel)
        is_base_kernel = is_kernel and not is_combo_kernel

        for op in self.operators:
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

    def expand_full_kernel(self, kernel: Kern, n_dims: int, base_kernels: List[str]):
        result = self.expand_single_kernel(kernel, n_dims, base_kernels)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in self.expand_full_kernel(operand, n_dims, base_kernels):
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

    def __init__(self, n_parents: int):
        super().__init__(n_parents)

    def initialize(self, kernel_families: List[str], n_kernels: int, n_dims: int):
        # use same initialization as CKS and BOMS
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels: List[AKSKernel], kernel_families: List[str], n_dims: int, verbose: bool = False):
        """Random expansion of nodes."""

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
            print('Expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n_parents={self.n_parents!r}, operators={self.operators!r})'

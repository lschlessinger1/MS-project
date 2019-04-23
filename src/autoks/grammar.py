from typing import List, Optional

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.hyperprior import Hyperpriors
from src.autoks.kernel import get_all_1d_kernels, create_1d_kernel, AKSKernel, remove_duplicate_kernels, \
    tree_to_kernel, pretty_print_aks_kernels, sort_kernel
from src.evalg.genprog import BinaryTreeGenerator, OnePointRecombinatorBase
from src.evalg.vary import PopulationOperator


class BaseGrammar:
    DEFAULT_OPERATORS: List[str] = ['+', '*']
    operators: List[str]

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 hyperpriors: Optional[Hyperpriors] = None):
        self.operators = BaseGrammar.DEFAULT_OPERATORS
        self.base_kernel_names = base_kernel_names
        self.n_dims = n_dims
        self.hyperpriors = hyperpriors

    def initialize(self) -> List[AKSKernel]:
        """Initialize kernels."""
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Get next round of candidate kernels from current kernels.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class EvolutionaryGrammar(BaseGrammar):
    population_operator: PopulationOperator
    initializer: Optional[BinaryTreeGenerator]
    n_init_trees: Optional[int]

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 population_operator,
                 initializer=None,
                 hyperpriors: Optional[Hyperpriors] = None,
                 n_init_trees=None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)
        self.population_operator = population_operator
        self.initializer = initializer
        self.n_init_trees = n_init_trees

    def initialize(self) -> List[AKSKernel]:
        """Initialize using initializer or same as CKS.

        :return:
        """
        if self.initializer is not None:
            n_init_trees = 10
            if self.n_init_trees is not None:
                self.n_init_trees = n_init_trees

            # Generate trees and then convert to GPy kernels, then to AKSKernels
            trees = [self.initializer.generate() for _ in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            aks_kernels = [AKSKernel(kernel) for kernel in kernels]
            return aks_kernels
        else:
            # Naive initialization of all SE_i and RQ_i (for every dimension).
            return CKSGrammar.all_1d_aks_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Perform crossover and mutation.

        :param seed_kernels: list of AKSKernels
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(seed_kernels, 'Seed')

        using_1_pt_cx = any([isinstance(v.operator, OnePointRecombinatorBase) for v in
                             self.population_operator.variators])
        if using_1_pt_cx:
            if verbose:
                print('Using one-point crossover. Sorting kernels.\n')
            # Sort trees if performing one-point crossover for alignment of trees.
            for aks_kernel in seed_kernels:
                aks_kernel.kernel = sort_kernel(aks_kernel.kernel)

        # Convert GPy kernels to BinaryTrees
        trees = [aks_kernel.to_binary_tree() for aks_kernel in seed_kernels]

        # Mutate/Crossover Trees
        offspring = self.population_operator.create_offspring(trees)

        # Convert Trees back to GPy kernels, then to AKSKernels
        kernels = [tree_to_kernel(tree) for tree in offspring]

        new_kernels = remove_duplicate_kernels(kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class BOMSGrammar(BaseGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)

    def initialize(self) -> List[AKSKernel]:
        """Initialize kernels according to number of dimensions.

        :return:
        """
        return CKSGrammar.all_1d_aks_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Greedy and exploratory expansion of kernels.

        :param seed_kernels: list of AKSKernels
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(seed_kernels, 'Seed')

        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        rw_kerns = self.random_walk_kernels()
        rw_kerns = remove_duplicate_kernels(rw_kerns)

        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        evaluated_kernels = [kernel for kernel in seed_kernels if kernel.evaluated]
        best_kern = sorted(evaluated_kernels, key=lambda x: x.score, reverse=True)[0]
        greedy_kerns = self.greedy_kernels(best_kern)
        greedy_kerns = remove_duplicate_kernels(greedy_kerns)

        new_kernels = rw_kerns + greedy_kerns
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def random_walk_kernels(self,
                            t_prob: float = 1 / 3.,
                            n_walks: int = 15) -> List[Kern]:
        """Geometric random walk kernels.

        :param t_prob: termination probability
        :param n_walks: number of random walks
        :return:
        """
        # geometric random walk
        max_depth = 10
        n_steps = np.random.geometric(p=t_prob, size=n_walks)
        n_steps[n_steps > max_depth] = max_depth

        rw_kernels = []
        for n in n_steps:
            # first expansion of empty kernel is all 1d kernels
            kernels = get_all_1d_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)
            random_kernel = np.random.choice(kernels)
            for i in range(1, n):
                kernels = CKSGrammar.expand_full_kernel(random_kernel, self.n_dims, self.base_kernel_names,
                                                        self.hyperpriors)
                random_kernel = np.random.choice(kernels)
                rw_kernels.append(random_kernel)

        return rw_kernels

    def greedy_kernels(self,
                       best_kernel: AKSKernel) -> List[Kern]:
        """Single expansion of CKS Grammar.

        :param best_kernel:
        :return:
        """
        new_kernels = CKSGrammar.expand_full_kernel(best_kernel.kernel, self.n_dims, self.base_kernel_names,
                                                    self.hyperpriors)

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)

    @staticmethod
    def get_base_kernel_names(n_dims: int) -> List[str]:
        if n_dims > 1:
            return ['SE', 'RQ']
        else:
            return ['SE', 'RQ', 'LIN', 'PER']

    @staticmethod
    def all_1d_aks_kernels(kernel_families: List[str],
                           n_dims: int,
                           hyperpriors: Optional[Hyperpriors] = None):
        kernels = get_all_1d_kernels(kernel_families, n_dims, hyperpriors=hyperpriors)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def initialize(self) -> List[AKSKernel]:
        """Initialize with all base kernel families applied to all input dimensions.

        :return:
        """
        return self.all_1d_aks_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Greedy expansion of nodes.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        # choose highest scoring kernel (using BIC) and expand it by applying all possible operators
        # CFG:
        # 1) Any subexpression S can be replaced with S + B, where B is any base kernel family.
        # 2) Any subexpression S can be replaced with S x B, where B is any base kernel family.
        # 3) Any base kernel B may be replaced with any other base kernel family B'
        if verbose:
            pretty_print_aks_kernels(seed_kernels, 'Seed')

        new_kernels = []
        for aks_kernel in seed_kernels:
            kernel = aks_kernel.kernel
            kernels_expanded = CKSGrammar.expand_full_kernel(kernel, self.n_dims, self.base_kernel_names,
                                                             self.hyperpriors)
            new_kernels += kernels_expanded

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    @staticmethod
    def expand_single_kernel(kernel: Kern,
                             n_dims: int,
                             base_kernels: List[str],
                             hyperpriors: Optional[Hyperpriors] = None) -> List[Kern]:
        """Expand a single kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param hyperpriors:
        :return:
        """
        is_kernel = isinstance(kernel, Kern)
        if not is_kernel:
            raise TypeError(f'Unknown kernel type {kernel.__class__.__name__}')

        kernels = []

        is_combo_kernel = isinstance(kernel, CombinationKernel)
        is_base_kernel = is_kernel and not is_combo_kernel

        for op in CKSGrammar.DEFAULT_OPERATORS:
            for d in range(n_dims):
                for base_kernel_name in base_kernels:
                    hyperprior = None if hyperpriors is None else hyperpriors[base_kernel_name]
                    if op == '+':
                        kernels.append(kernel + create_1d_kernel(base_kernel_name, d, hyperpriors=hyperprior))
                    elif op == '*':
                        kernels.append(kernel * create_1d_kernel(base_kernel_name, d, hyperpriors=hyperprior))
                    else:
                        raise ValueError(f'Unknown operation {op}')
        for base_kernel_name in base_kernels:
            if is_base_kernel:
                hyperprior = None if hyperpriors is None else hyperpriors[base_kernel_name]
                kernels.append(create_1d_kernel(base_kernel_name, kernel.active_dims[0], hyperpriors=hyperprior))
        return kernels

    @staticmethod
    def expand_full_kernel(kernel: Kern,
                           n_dims: int,
                           base_kernels: List[str],
                           hyperpriors: Optional[Hyperpriors] = None) -> List[Kern]:
        """Expand full kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param hyperpriors:
        :return:
        """
        result = CKSGrammar.expand_single_kernel(kernel, n_dims, base_kernels, hyperpriors)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in CKSGrammar.expand_full_kernel(operand, n_dims, base_kernels, hyperpriors):
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
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class RandomGrammar(BaseGrammar):
    """Random grammar randomly expands nodes

    """

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)

    def initialize(self) -> List[AKSKernel]:
        """Same initialization as CKS and BOMS

        :return:
        """
        return CKSGrammar.all_1d_aks_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Random expansion of nodes.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(seed_kernels, 'Seed')

        new_kernels = []

        # For each kernel, select a kernel from one step of a CKS expansion uniformly at random.
        for aks_kernel in seed_kernels:
            k = aks_kernel.kernel
            expansion = CKSGrammar.expand_full_kernel(k, self.n_dims, self.base_kernel_names)
            k = np.random.choice(expansion)
            new_kernels.append(k)

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'

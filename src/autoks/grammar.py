from typing import List, Optional

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.hyperprior import Hyperpriors, BOMSHyperpriors
from src.autoks.kernel import get_all_1d_kernels, create_1d_kernel, AKSKernel, remove_duplicate_kernels, tree_to_kernel, \
    pretty_print_aks_kernels
from src.evalg.genprog import BinaryTreeGenerator
from src.evalg.vary import PopulationOperator


class BaseGrammar:
    DEFAULT_OPERATORS: List[str] = ['+', '*']
    operators: List[str]

    def __init__(self):
        self.operators = BaseGrammar.DEFAULT_OPERATORS

    def initialize(self,
                   kernel_families: List[str],
                   n_dims: int,
                   hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Initialize kernels.

        :param kernel_families:
        :param n_dims:
        :param hyperpriors:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self,
               kernels: List[AKSKernel],
               kernel_families: List[str],
               n_dims: int,
               verbose: bool = False,
               hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Get next round of candidate kernels from current kernels.

        :param kernels:
        :param kernel_families:
        :param n_dims: number of dimensions
        :param verbose:
        :param hyperpriors:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class EvolutionaryGrammar(BaseGrammar):
    population_operator: PopulationOperator
    initializer: Optional[BinaryTreeGenerator]
    n_init_trees: Optional[int]

    def __init__(self, population_operator, initializer=None, n_init_trees=None):
        super().__init__()
        self.population_operator = population_operator
        self.initializer = initializer
        self.n_init_trees = n_init_trees

    def initialize(self,
                   kernel_families: List[str],
                   n_dims: int,
                   hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Initialize using initializer or same as CKS.

        :param kernel_families:
        :param n_dims:
        :param hyperpriors:
        :return:
        """
        if self.initializer is not None:
            n_init_trees = 10
            if self.n_init_trees is not None:
                self.n_init_trees = n_init_trees

            # Generate trees and then convert to GPy kernels, then to AKSKernels
            trees = [self.initializer.generate() for i in range(self.n_init_trees)]
            kernels = [tree_to_kernel(tree) for tree in trees]
            aks_kernels = [AKSKernel(kernel) for kernel in kernels]
            return aks_kernels
        else:
            # Naive initialization of all SE_i and RQ_i (for every dimension).
            return CKSGrammar.all_1d_aks_kernels(kernel_families, n_dims, hyperpriors)

    def expand(self,
               aks_kernels: List[AKSKernel],
               kernel_families: List[str],
               n_dims: int,
               verbose: bool = False,
               hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Perform crossover and mutation.

        :param aks_kernels: list of AKSKernels
        :param kernel_families: base kernels
        :param n_dims:
        :param verbose:
        :param hyperpriors:
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(aks_kernels, 'Seed')

        # Convert GPy kernels to BinaryTrees
        trees = [aks_kernel.to_binary_tree() for aks_kernel in aks_kernels]

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

    def __init__(self):
        super().__init__()

    def initialize(self,
                   kernel_families: List[str],
                   n_dims: int,
                   hyperpriors: Optional[BOMSHyperpriors] = None) -> List[AKSKernel]:
        """Initialize kernels according to number of dimensions.

        :param kernel_families:
        :param n_dims:
        :param hyperpriors
        :return:
        """
        return CKSGrammar.all_1d_aks_kernels(kernel_families, n_dims, hyperpriors)

    def expand(self,
               aks_kernels: List[AKSKernel],
               kernel_families: List[str],
               n_dims: int,
               verbose: bool = False,
               hyperpriors: Optional[BOMSHyperpriors] = None) -> List[AKSKernel]:
        """Greedy and exploratory expansion of kernels.

        :param aks_kernels: list of AKSKernels
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :param hyperpriors
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(aks_kernels, 'Seed')

        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        rw_kerns = BOMSGrammar.random_walk_kernels(n_dims, kernel_families, hyperpriors=hyperpriors)
        rw_kerns = remove_duplicate_kernels(rw_kerns)

        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        evaluated_kernels = [kernel for kernel in aks_kernels if kernel.evaluated]
        best_kern = sorted(evaluated_kernels, key=lambda x: x.score, reverse=True)[0]
        greedy_kerns = BOMSGrammar.greedy_kernels(best_kern, n_dims, kernel_families, hyperpriors=hyperpriors)
        greedy_kerns = remove_duplicate_kernels(greedy_kerns)

        new_kernels = rw_kerns + greedy_kerns
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    @staticmethod
    def random_walk_kernels(n_dims: int,
                            base_kernels: List[str],
                            t_prob: float = 1 / 3.,
                            n_walks: int = 15,
                            hyperpriors: Optional[BOMSHyperpriors] = None) -> List[Kern]:
        """Geometric random walk kernels.

        :param n_dims:
        :param base_kernels:
        :param t_prob: termination probability
        :param n_walks: number of random walks
        :param hyperpriors
        :return:
        """
        # geometric random walk
        max_depth = 10
        n_steps = np.random.geometric(p=t_prob, size=n_walks)
        n_steps[n_steps > max_depth] = max_depth

        rw_kernels = []
        for n in n_steps:
            # first expansion of empty kernel is all 1d kernels
            kernels = get_all_1d_kernels(base_kernels, n_dims, hyperpriors)
            random_kernel = np.random.choice(kernels)
            for i in range(1, n):
                kernels = CKSGrammar.expand_full_kernel(random_kernel, n_dims, base_kernels,
                                                        BaseGrammar.DEFAULT_OPERATORS, hyperpriors)
                random_kernel = np.random.choice(kernels)
                rw_kernels.append(random_kernel)

        return rw_kernels

    @staticmethod
    def greedy_kernels(best_kernel: AKSKernel,
                       n_dims: int,
                       base_kernels: List[str],
                       hyperpriors: Optional[BOMSHyperpriors] = None) -> List[Kern]:
        """Single expansion of CKS Grammar.

        :param best_kernel:
        :param n_dims:
        :param base_kernels:
        :param hyperpriors
        :return:
        """
        new_kernels = CKSGrammar.expand_full_kernel(best_kernel.kernel, n_dims, base_kernels,
                                                    CKSGrammar.DEFAULT_OPERATORS, hyperpriors)

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_base_kernels(n_dims: int) -> List[str]:
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

    def initialize(self,
                   kernel_families: List[str],
                   n_dims: int,
                   hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Initialize with all base kernel families applied to all input dimensions.

        :param kernel_families:
        :param n_dims:
        :param hyperpriors:
        :return:
        """
        return self.all_1d_aks_kernels(kernel_families, n_dims, hyperpriors)

    def expand(self,
               aks_kernels: List[AKSKernel],
               kernel_families: List[str],
               n_dims: int,
               verbose: bool = False,
               hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Greedy expansion of nodes.

        :param aks_kernels:
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :param hyperpriors:
        :return:
        """
        # choose highest scoring kernel (using BIC) and expand it by applying all possible operators
        # CFG:
        # 1) Any subexpression S can be replaced with S + B, where B is any base kernel family.
        # 2) Any subexpression S can be replaced with S x B, where B is any base kernel family.
        # 3) Any base kernel B may be replaced with any other base kernel family B'
        if verbose:
            pretty_print_aks_kernels(aks_kernels, 'Seed')

        new_kernels = []
        for aks_kernel in aks_kernels:
            kernel = aks_kernel.kernel
            kernels_expanded = CKSGrammar.expand_full_kernel(kernel, n_dims, kernel_families, self.operators,
                                                             hyperpriors)
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
                             operators: List[str],
                             hyperpriors: Optional[Hyperpriors] = None) -> List[Kern]:
        """Expand a single kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param operators:
        :param hyperpriors:
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
                           operators: List[str],
                           hyperpriors: Optional[Hyperpriors] = None) -> List[Kern]:
        """Expand full kernel.

        :param kernel:
        :param n_dims:
        :param base_kernels:
        :param operators:
        :param hyperpriors:
        :return:
        """
        result = CKSGrammar.expand_single_kernel(kernel, n_dims, base_kernels, operators, hyperpriors)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in CKSGrammar.expand_full_kernel(operand, n_dims, base_kernels, operators, hyperpriors):
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

    def __init__(self):
        super().__init__()

    def initialize(self,
                   kernel_families: List[str],
                   n_dims: int,
                   hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Same initialization as CKS and BOMS

        :param kernel_families:
        :param n_dims:
        :param hyperpriors
        :return:
        """
        return CKSGrammar.all_1d_aks_kernels(kernel_families, n_dims, hyperpriors)

    def expand(self,
               aks_kernels: List[AKSKernel],
               kernel_families: List[str],
               n_dims: int,
               verbose: bool = False,
               hyperpriors: Optional[Hyperpriors] = None) -> List[AKSKernel]:
        """Random expansion of nodes.

        :param aks_kernels:
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :param hyperpriors
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(aks_kernels, 'Seed')

        new_kernels = []

        # for each kernel, randomly add or multiply a random 1D kernel
        for aks_kernel in aks_kernels:
            k = aks_kernel.kernel
            all_1d_kernels = get_all_1d_kernels(kernel_families, n_dims, hyperpriors)

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
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r})'

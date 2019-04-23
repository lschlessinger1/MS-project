from typing import List, Optional

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.hyperprior import Hyperpriors, boms_hyperpriors
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
        self.base_kernels = get_all_1d_kernels(self.base_kernel_names, self.n_dims, hyperpriors=self.hyperpriors)

    def initialize(self) -> List[AKSKernel]:
        """Initialize kernels."""
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self,
               seed_kernels: List[AKSKernel],
               verbose: bool = False) -> List[AKSKernel]:
        """Expand seed kernels.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def get_candidates(self,
                       seed_kernels: List[AKSKernel],
                       verbose: bool = False) -> List[AKSKernel]:
        """Get next round of candidate kernels from current kernels.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        return self.expand(seed_kernels, verbose=verbose)

    @staticmethod
    def _kernels_to_aks_kernels(kernels: List[Kern]) -> List[AKSKernel]:
        return [AKSKernel(kernel) for kernel in kernels]

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r})'


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
            aks_kernels = self._kernels_to_aks_kernels(kernels)
            return aks_kernels
        else:
            # Naive initialization of all SE_i and RQ_i (for every dimension).
            return self._kernels_to_aks_kernels(self.base_kernels)

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
        new_kernels = self._kernels_to_aks_kernels(new_kernels)

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r},' \
            f'population_operator={self.population_operator!r}, initializer={self.initializer!r}, ' \
            f'n_init_trees={self.n_init_trees!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self,
                 n_dims: int,
                 base_kernel_names: List[str] = None,
                 hyperpriors: Optional[Hyperpriors] = None):
        if base_kernel_names is None:
            base_kernel_names = self.get_base_kernel_names(n_dims)
        super().__init__(base_kernel_names, n_dims, hyperpriors)

    @staticmethod
    def get_base_kernel_names(n_dims: int) -> List[str]:
        if n_dims > 1:
            return ['SE', 'RQ']
        else:
            return ['SE', 'RQ', 'LIN', 'PER']

    def initialize(self) -> List[AKSKernel]:
        """Initialize with all base kernel families applied to all input dimensions.

        :return:
        """
        return self._kernels_to_aks_kernels(self.base_kernels)

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
            kernels_expanded = self.expand_full_kernel(kernel)
            new_kernels += kernels_expanded

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = self._kernels_to_aks_kernels(new_kernels)

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def expand_single_kernel(self, kernel: Kern) -> List[Kern]:
        """Expand a single kernel.

        :param kernel:
        :return:
        """
        is_kernel = isinstance(kernel, Kern)
        if not is_kernel:
            raise TypeError(f'Unknown kernel type {kernel.__class__.__name__}')

        kernels = []

        is_combo_kernel = isinstance(kernel, CombinationKernel)
        is_base_kernel = is_kernel and not is_combo_kernel

        for op in self.operators:
            for dim in range(self.n_dims):
                for base_kernel_name in self.base_kernel_names:
                    hyperprior = None if self.hyperpriors is None else self.hyperpriors[base_kernel_name]
                    if op == '+':
                        kernels.append(kernel + create_1d_kernel(base_kernel_name, dim, hyperpriors=hyperprior))
                    elif op == '*':
                        kernels.append(kernel * create_1d_kernel(base_kernel_name, dim, hyperpriors=hyperprior))
                    else:
                        raise ValueError(f'Unknown operation {op}')
        for base_kernel_name in self.base_kernel_names:
            if is_base_kernel:
                hyperprior = None if self.hyperpriors is None else self.hyperpriors[base_kernel_name]
                kernels.append(create_1d_kernel(base_kernel_name, kernel.active_dims[0], hyperpriors=hyperprior))
        return kernels

    def expand_full_kernel(self, kernel: Kern) -> List[Kern]:
        """Expand full kernel.

        :param kernel:
        :return:
        """
        result = self.expand_single_kernel(kernel)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in self.expand_full_kernel(operand):
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


class BOMSGrammar(CKSGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """
    random_walk_geometric_dist_parameter: float
    number_of_top_k_best: int
    number_of_random_walks: int

    def __init__(self, base_kernel_names: List[str], n_dims: int, hyperpriors: Optional[Hyperpriors] = None):

        if hyperpriors is None:
            hyperpriors = boms_hyperpriors()

        super().__init__(n_dims, base_kernel_names, hyperpriors)

        self.random_walk_geometric_dist_parameter = 1 / 3  # termination probability
        self.number_of_top_k_best = 3
        self.number_of_random_walks = 15

    def initialize(self) -> List[AKSKernel]:
        """Initialize kernels according to number of dimensions.

        :return:
        """
        return self._kernels_to_aks_kernels(self.base_kernels)

    def get_candidates(self,
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
        new_kernels = self._kernels_to_aks_kernels(new_kernels)

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

    def random_walk_kernels(self) -> List[Kern]:
        """Geometric random walk kernels.

        :return:
        """
        # geometric random walk
        max_depth = 10
        n_steps = np.random.geometric(p=self.random_walk_geometric_dist_parameter, size=self.number_of_random_walks)
        n_steps[n_steps > max_depth] = max_depth

        rw_kernels = []
        for n in n_steps:
            # first expansion of empty kernel is all 1d kernels
            kernels = get_all_1d_kernels(self.base_kernel_names, self.n_dims, self.hyperpriors)
            random_kernel = np.random.choice(kernels)
            for i in range(1, n):
                kernels = self.expand_full_kernel(random_kernel)
                random_kernel = np.random.choice(kernels)
                rw_kernels.append(random_kernel)

        return rw_kernels

    def greedy_kernels(self,
                       best_kernel: AKSKernel) -> List[Kern]:
        """Single expansion of CKS Grammar.

        :param best_kernel:
        :return:
        """
        new_kernels = self.expand_full_kernel(best_kernel.kernel)

        return new_kernels


class RandomGrammar(CKSGrammar):
    """Random grammar randomly expands nodes using a CKS expansion"""

    def __init__(self,
                 n_dims: int,
                 base_kernel_names: List[str] = None,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(n_dims, base_kernel_names, hyperpriors)

        self.max_n_kernels = 10

    def get_candidates(self,
                       seed_kernels: List[AKSKernel],
                       verbose: bool = False) -> List[AKSKernel]:
        """Random expansion of nodes.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_aks_kernels(seed_kernels, 'Seed')

        # Select kernels from one step of a CKS expansion uniformly at random without replacement.
        cks_expansion = self.expand(seed_kernels, verbose=False)
        n_kernels = min(self.max_n_kernels, min(len(cks_expansion), self.max_n_kernels))
        new_kernels = list(np.random.choice(cks_expansion, size=n_kernels, replace=False).tolist())

        if verbose:
            pretty_print_aks_kernels(new_kernels, 'Newly expanded')

        return new_kernels

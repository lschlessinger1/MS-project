import warnings
from typing import List, Optional

import numpy as np
from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from src.autoks.hyperprior import Hyperpriors, boms_hyperpriors
from src.autoks.kernel import get_all_1d_kernels, GPModel, remove_duplicate_kernels, \
    tree_to_kernel, pretty_print_gp_models, sort_kernel
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

    def initialize(self) -> List[GPModel]:
        """Initialize kernels."""
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self,
               seed_kernels: List[GPModel],
               verbose: bool = False) -> List[GPModel]:
        """Expand seed kernels.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def get_candidates(self,
                       seed_kernels: List[GPModel],
                       verbose: bool = False) -> List[GPModel]:
        """Get next round of candidate kernels from current kernels.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        return self.expand(seed_kernels, verbose=verbose)

    @staticmethod
    def _kernels_to_gp_models(kernels: List[Kern]) -> List[GPModel]:
        return [GPModel(kernel) for kernel in kernels]

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

    def initialize(self) -> List[GPModel]:
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
            aks_kernels = self._kernels_to_gp_models(kernels)
            return aks_kernels
        else:
            # Naive initialization of all SE_i and RQ_i (for every dimension).
            return self._kernels_to_gp_models(self.base_kernels)

    def expand(self,
               seed_kernels: List[GPModel],
               verbose: bool = False) -> List[GPModel]:
        """Perform crossover and mutation.

        :param seed_kernels: list of AKSKernels
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_gp_models(seed_kernels, 'Seed')

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
        new_kernels = self._kernels_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

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

    def initialize(self) -> List[GPModel]:
        """Initialize with all base kernel families applied to all input dimensions.

        :return:
        """
        return self._kernels_to_gp_models(self.base_kernels)

    def expand(self,
               seed_kernels: List[GPModel],
               verbose: bool = False) -> List[GPModel]:
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
            pretty_print_gp_models(seed_kernels, 'Seed')

        new_kernels = []
        for aks_kernel in seed_kernels:
            new_kernels += self.expand_full_kernel(aks_kernel.kernel)

        new_kernels = [sort_kernel(kernel) for kernel in new_kernels]
        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = self._kernels_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

    def expand_single_kernel(self, seed_kernel: Kern) -> List[Kern]:
        """Expand a seed kernel according to the CKS grammar.

        :param seed_kernel:
        :return:
        """
        new_kernels = []

        for base_kernel in self.base_kernels:
            new_kernels.append(seed_kernel + base_kernel)
            new_kernels.append(seed_kernel * base_kernel)

        return new_kernels

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

    def expand_full_brute_force(self,
                                level: int,
                                max_number_of_models: int) -> List[Kern]:
        """Enumerate all kernels in CKS grammar up to a depth with a size limit."""
        if level >= 4:
            warnings.warn('This is a brute-force implementation, use it for level < 4')

        current_kernels = self.base_kernels
        if level == 0:
            return current_kernels

        def remove_duplicates(new_kerns: List[Kern], all_kerns: List[Kern]):
            unique_kernels = []
            for new_kernel in new_kerns:
                # TODO: this should be using a wrapper of Kern, not GPModel.
                symbolic_expr_new_kern = GPModel(new_kernel).symbolic_expr
                repeated = False
                for kern in all_kerns:
                    symbolic_expr_kern = GPModel(kern).symbolic_expr
                    kerns_equal = symbolic_expr_new_kern == symbolic_expr_kern
                    if kerns_equal:
                        repeated = True
                        break
                if not repeated:
                    unique_kernels.append(new_kernel)
            return unique_kernels

        all_kernels = []

        while level > 0:
            this_level = []
            number_of_models = len(all_kernels)
            for current_kernel in current_kernels:
                new_kernels = self.expand_single_kernel(current_kernel)
                unique_new_kernels = remove_duplicates(new_kernels, this_level)
                this_level += unique_new_kernels
                number_of_models = number_of_models + len(unique_new_kernels)
                if number_of_models > max_number_of_models:
                    all_kernels += this_level
                    all_kernels = all_kernels[:max_number_of_models]
                    return all_kernels
            current_kernels = this_level
            all_kernels += this_level
            level -= 1
        return all_kernels


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

    def initialize(self) -> List[GPModel]:
        """Initialize kernels according to number of dimensions.

        :return:
        """
        initial_level_depth = 2
        max_number_of_initial_models = 500
        initial_candidates = self.expand_full_brute_force(initial_level_depth, max_number_of_initial_models)
        return self._kernels_to_gp_models(initial_candidates)

    def get_candidates(self,
                       seed_kernels: List[GPModel],
                       verbose: bool = False) -> List[GPModel]:
        """Greedy and exploratory expansion of kernels.

        :param seed_kernels: list of AKSKernels
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_gp_models(seed_kernels, 'Seed')

        # Exploration
        total_num_walks = self.number_of_random_walks
        candidates_random = self.expand_random(total_num_walks)

        # Exploitation
        evaluated_kernels = [kernel for kernel in seed_kernels if kernel.evaluated]
        fitness_score = [k.score for k in evaluated_kernels]
        candidates_best = self.expand_best(evaluated_kernels, fitness_score)

        # Concatenate
        candidates = candidates_best + candidates_random
        new_kernels = self._kernels_to_gp_models(candidates)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

    def expand_random(self, total_num_walks: int) -> List[Kern]:
        """Geometric random walk kernels.

        :return:
        """
        parameter = self.random_walk_geometric_dist_parameter
        depths = np.random.geometric(parameter, size=total_num_walks)
        new_kernels = []
        for depth in depths:
            frontier = self.base_kernels
            new_kernel = np.random.choice(frontier)
            for i in range(depth - 1):
                new_kernel = np.random.choice(frontier)
                frontier = self.expand_single_kernel(new_kernel)
            new_kernels.append(new_kernel)

        return new_kernels

    def expand_best(self,
                    selected_models: List[GPModel],
                    fitness_score: List[float]) -> List[Kern]:
        """Single expansion of CKS Grammar.

        :param selected_models:
        :param fitness_score:
        :return:
        """
        new_kernels = []
        num_exploit_top = self.number_of_top_k_best
        if len(fitness_score) < 2:
            return new_kernels

        indices = np.argsort(fitness_score)[::-1]
        last_index = min(num_exploit_top, len(fitness_score))
        indices = indices[:last_index]

        models_with_highest_score = [selected_models[i] for i in indices]
        for model in models_with_highest_score:
            kernel_to_expand = model.kernel
            new_kernel_list = self.expand_single_kernel(kernel_to_expand)
            new_kernels += new_kernel_list

        return new_kernels


class RandomGrammar(CKSGrammar):
    """Random grammar randomly expands nodes using a CKS expansion"""

    def __init__(self,
                 n_dims: int,
                 base_kernel_names: List[str] = None,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(n_dims, base_kernel_names, hyperpriors)

        self.max_n_kernels = 1

    def get_candidates(self,
                       seed_kernels: List[GPModel],
                       verbose: bool = False) -> List[GPModel]:
        """Random expansion of nodes.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        if verbose:
            pretty_print_gp_models(seed_kernels, 'Seed')

        # Select kernels from one step of a CKS expansion uniformly at random without replacement.
        cks_expansion = []
        for seed_kernel in seed_kernels:
            cks_expansion += self.expand_single_kernel(seed_kernel.kernel)

        n_kernels = min(self.max_n_kernels, min(len(cks_expansion), self.max_n_kernels))
        new_kernels = list(np.random.choice(cks_expansion, size=n_kernels, replace=False).tolist())

        new_kernels = self._kernels_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

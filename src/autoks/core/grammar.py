import warnings
from typing import List, Optional

import numpy as np

from src.autoks.backend.kernel import get_all_1d_kernels, sort_kernel
from src.autoks.core.covariance import Covariance, remove_duplicate_kernels
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.hyperprior import Hyperpriors, boms_hyperpriors
from src.autoks.core.kernel_encoding import tree_to_kernel
from src.evalg.genprog import OnePointRecombinatorBase
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
        raw_kernels = get_all_1d_kernels(self.base_kernel_names, self.n_dims, hyperpriors=self.hyperpriors)
        self.base_kernels = [Covariance(kernel) for kernel in raw_kernels]

    def expand(self,
               seed_kernels: List[GPModel],
               verbose: bool = False) -> List[GPModel]:
        """Expand seed gp_models.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def get_candidates(self,
                       seed_kernels: List[GPModel],
                       verbose: bool = False) -> List[GPModel]:
        """Get next round of candidate gp_models from current gp_models.

        :param seed_kernels:
        :param verbose:
        :return:
        """
        return self.expand(seed_kernels, verbose=verbose)

    @staticmethod
    def _covariances_to_gp_models(covariances: List[Covariance]) -> List[GPModel]:
        return [GPModel(covariance) for covariance in covariances]

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r})'


class EvolutionaryGrammar(BaseGrammar):
    population_operator: PopulationOperator

    def __init__(self,
                 base_kernel_names: List[str],
                 n_dims: int,
                 population_operator,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)
        self.population_operator = population_operator

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
                print('Using one-point crossover. Sorting gp_models.\n')
            # Sort trees if performing one-point crossover for alignment of trees.
            for seed_model in seed_kernels:
                seed_model.covariance.raw_kernel = seed_model.covariance.canonical()

        # Convert GPy gp_models to BinaryTrees
        trees = [gp_model.covariance.to_binary_tree() for gp_model in seed_kernels]

        # Mutate/Crossover Trees
        offspring = self.population_operator.create_offspring(trees)

        # Convert Trees back to GPy gp_models, then to GPModels
        kernels = [tree_to_kernel(tree) for tree in offspring]

        covariances = [Covariance(k) for k in kernels]

        new_kernels = remove_duplicate_kernels(covariances)
        new_kernels = self._covariances_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r},' \
            f'population_operator={self.population_operator!r})'


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
        for gp_model in seed_kernels:
            new_kernels += self.expand_full_kernel(gp_model.covariance)

        new_kernels = [sort_kernel(kernel) for kernel in new_kernels]
        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = self._covariances_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

    def expand_single_kernel(self, seed_kernel: Covariance) -> List[Covariance]:
        """Expand a seed kernel according to the CKS grammar.

        :param seed_kernel:
        :return:
        """
        new_kernels = []

        for base_kernel in self.base_kernels:
            new_kernels.append(seed_kernel + base_kernel)
            new_kernels.append(seed_kernel * base_kernel)

        return new_kernels

    def expand_full_kernel(self, kernel: Covariance) -> List[Covariance]:
        """Expand full kernel.

        :param kernel:
        :return:
        """
        result = self.expand_single_kernel(kernel)
        if kernel is None:
            pass
        elif not kernel.is_base():
            for i, operand in enumerate(kernel.raw_kernel.parts):
                covariance_operand = Covariance(operand)
                for e in self.expand_full_kernel(covariance_operand):
                    new_operands = kernel.raw_kernel.parts[:i] + [e.raw_kernel] + kernel.raw_kernel.parts[i + 1:]
                    new_operands = [op.copy() for op in new_operands]
                    if kernel.is_prod():
                        prod_kern = new_operands[0]
                        for part in new_operands[1:]:
                            prod_kern *= part
                        result.append(Covariance(prod_kern))
                    elif kernel.is_sum():
                        prod_kern = new_operands[0]
                        for part in new_operands[1:]:
                            prod_kern += part
                        result.append(Covariance(prod_kern))
                    else:
                        raise TypeError(f'Unknown combination kernel class {kernel.__class__.__name__}')

        return result

    def expand_full_brute_force(self,
                                level: int,
                                max_number_of_models: int) -> List[Covariance]:
        """Enumerate all gp_models in CKS grammar up to a depth with a size limit."""
        if level >= 4:
            warnings.warn('This is a brute-force implementation, use it for level < 4')

        current_kernels = self.base_kernels
        if level == 0:
            return current_kernels

        def remove_duplicates(new_kerns: List[Covariance], all_kerns: List[Covariance]):
            unique_kernels = []
            for new_kernel in new_kerns:
                repeated = False
                for kern in all_kerns:
                    if new_kernel.symbolic_expanded_equals(kern):
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

    def get_candidates(self,
                       seed_kernels: List[GPModel],
                       verbose: bool = False) -> List[GPModel]:
        """Greedy and exploratory expansion of gp_models.

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
        new_kernels = self._covariances_to_gp_models(candidates)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

    def expand_random(self, total_num_walks: int) -> List[Covariance]:
        """Geometric random walk gp_models.

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
                    fitness_score: List[float]) -> List[Covariance]:
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
            kernel_to_expand = model.covariance
            new_kernel_list = self.expand_single_kernel(kernel_to_expand)
            new_kernels += new_kernel_list

        return new_kernels


class RandomGrammar(BOMSGrammar):
    """Random grammar randomly expands nodes using a CKS expansion"""

    def __init__(self,
                 n_dims: int,
                 base_kernel_names: List[str] = None,
                 hyperpriors: Optional[Hyperpriors] = None):
        super().__init__(base_kernel_names, n_dims, hyperpriors)

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

        new_kernels = self.expand_random(self.number_of_random_walks)
        new_kernels = self._covariances_to_gp_models(new_kernels)

        if verbose:
            pretty_print_gp_models(new_kernels, 'Newly expanded')

        return new_kernels

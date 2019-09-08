import warnings
from typing import List, Optional, ClassVar

import numpy as np

from src.autoks.backend.kernel import get_all_1d_kernels, sort_kernel
from src.autoks.core.covariance import Covariance, remove_duplicate_kernels, pretty_print_covariances
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.hyperprior import boms_hyperpriors, HyperpriorMap
from src.autoks.core.kernel_encoding import tree_to_kernel
from src.evalg.genprog.crossover import OnePointRecombinatorBase
from src.evalg.serialization import Serializable
from src.evalg.vary import PopulationOperator


class BaseGrammar(Serializable):
    DEFAULT_OPERATORS: ClassVar[List[str]] = ['+', '*']
    operators: List[str]

    def __init__(self,
                 base_kernel_names: Optional[List[str]],
                 hyperpriors: Optional[HyperpriorMap] = None):
        self.operators = BaseGrammar.DEFAULT_OPERATORS
        self.base_kernel_names = base_kernel_names
        self.hyperpriors = hyperpriors or HyperpriorMap()
        self.n_dims = None
        self.base_kernels = None
        self.built = False

    def expand(self,
               seed_models: List[GPModel],
               verbose: int = 0) -> List[Covariance]:
        """Expand seed gp_models.

        :param seed_models: List of GP Models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def get_candidates(self,
                       seed_models: List[GPModel],
                       verbose: int = 0) -> List[Covariance]:
        """Get next round of candidate covariances from current GP models.

        :param seed_models: List of GP models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        if not self.built:
            raise RuntimeError('You must build a grammar before '
                               'getting candidates. '
                               'Use `grammar.build(n_dims)`.')

        if verbose == 3:
            pretty_print_gp_models(seed_models, 'Seed')

        new_covariances = self.expand(seed_models, verbose=verbose)

        if verbose == 3:
            pretty_print_covariances(new_covariances, 'Newly expanded')

        return new_covariances

    def build(self, n_dims: int):
        """Set the base kernels."""
        if self.base_kernel_names is None:
            self.base_kernel_names = self._get_default_base_kernel_names(n_dims)
        self.n_dims = n_dims
        raw_kernels = get_all_1d_kernels(self.base_kernel_names, self.n_dims, hyperpriors=self.hyperpriors.prior_map)
        self.base_kernels = [Covariance(kernel) for kernel in raw_kernels]
        self.built = True

    @staticmethod
    def _get_default_base_kernel_names(n_dims: int) -> List[str]:
        return ['SE', 'RQ']

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["operators"] = self.operators
        input_dict["base_kernel_names"] = self.base_kernel_names
        input_dict["hyperpriors"] = self.hyperpriors.to_dict()
        input_dict["n_dims"] = self.n_dims
        input_dict["base_kernels"] = None if self.base_kernels is None else [k.to_dict() for k in self.base_kernels]
        input_dict["built"] = self.built
        return input_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        operators = input_dict.pop('operators')
        n_dims = input_dict.pop('n_dims')
        base_kernels = input_dict.pop('base_kernels')
        built = input_dict.pop('built')

        grammar = super()._build_from_input_dict(input_dict)

        grammar.operators = operators
        grammar.n_dims = n_dims
        grammar.base_kernels = None if base_kernels is None else [Covariance.from_dict(k) for k in base_kernels]
        grammar.built = built

        return grammar

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        input_dict['hyperpriors'] = HyperpriorMap.from_dict(input_dict['hyperpriors'])
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r})'


class EvolutionaryGrammar(BaseGrammar):

    def __init__(self,
                 population_operator: PopulationOperator,
                 base_kernel_names: Optional[List[str]] = None,
                 hyperpriors: Optional[HyperpriorMap] = None):
        super().__init__(base_kernel_names, hyperpriors)
        self.population_operator = population_operator

    def expand(self,
               seed_models: List[GPModel],
               verbose: int = 0) -> List[Covariance]:
        """Perform crossover and mutation.

        :param seed_models: List of GP models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        using_1_pt_cx = any([isinstance(v.operator, OnePointRecombinatorBase) for v in
                             self.population_operator.variators])
        if using_1_pt_cx:
            if verbose == 3:
                print('Using one-point crossover. Sorting gp_models.\n')
            # Sort trees if performing one-point crossover for alignment of trees.
            for seed_model in seed_models:
                seed_model.covariance.raw_kernel = seed_model.covariance.canonical()

        # Convert GP models to binary trees.
        trees = [gp_model.covariance.to_binary_tree() for gp_model in seed_models]

        # Crossover and mutate trees.
        operands = [cov.raw_kernel for cov in self.base_kernels]
        offspring = self.population_operator.create_offspring(trees, operators=self.operators, operands=operands)

        # Convert Trees back to GPy kernels, then to covariances.
        kernels = [tree_to_kernel(tree) for tree in offspring]

        covariances = [Covariance(k) for k in kernels]

        new_covariances = remove_duplicate_kernels(covariances)

        # Because GPy doesn't serialize hyperparameter priors in `to_dict`, they must be reset.
        if self.hyperpriors:
            for cov in new_covariances:
                cov.set_hyperpriors(self.hyperpriors)

        return new_covariances

    @staticmethod
    def _get_default_base_kernel_names(n_dims: int) -> List[str]:
        return CKSGrammar.default_base_kernel_names(n_dims)

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict["population_operator"] = self.population_operator.to_dict()
        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict):
        input_dict = super()._format_input_dict(input_dict)
        input_dict['population_operator'] = PopulationOperator.from_dict(input_dict['population_operator'])
        return input_dict

    def __repr__(self):
        return f'{self.__class__.__name__}('f'operators={self.operators!r}, ' \
            f'base_kernel_names={self.base_kernel_names!r}, n_dims={self.n_dims!r}, hyperpriors={self.hyperpriors!r},' \
            f'population_operator={self.population_operator!r})'


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self,
                 base_kernel_names: Optional[List[str]] = None,
                 hyperpriors: Optional[HyperpriorMap] = None):
        super().__init__(base_kernel_names, hyperpriors)

    @staticmethod
    def default_base_kernel_names(n_dims: int) -> List[str]:
        """Get the names of the kernel families to use according to the dimension."""
        if n_dims > 1:
            return ['SE', 'RQ']
        else:
            return ['SE', 'RQ', 'LIN', 'PER']

    @staticmethod
    def _get_default_base_kernel_names(n_dims: int) -> List[str]:
        return CKSGrammar.default_base_kernel_names(n_dims)

    def expand(self,
               seed_models: List[GPModel],
               verbose: int = 0) -> List[Covariance]:
        """Greedy expansion of nodes.

        Choose highest scoring kernel and expand it by applying all possible operators.
        Context-free grammar rules:
        1) Any subexpression S can be replaced with S + B, where B is any base kernel family.
        2) Any subexpression S can be replaced with S x B, where B is any base kernel family.
        3) Any base kernel B may be replaced with any other base kernel family B'

        :param seed_models: List of GP models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        new_covariances = []
        for gp_model in seed_models:
            new_covariances += self.expand_full_kernel(gp_model.covariance)

        new_covariances = [sort_kernel(kernel) for kernel in new_covariances]
        new_covariances = remove_duplicate_kernels(new_covariances)

        return new_covariances

    def expand_single_kernel(self, seed_kernel: Covariance) -> List[Covariance]:
        """Expand a seed kernel according to the CKS grammar.

        :param seed_kernel: Covariance to expand.
        :return:
        """
        new_covariances = []

        for base_kernel in self.base_kernels:
            new_covariances.append(seed_kernel + base_kernel)
            new_covariances.append(seed_kernel * base_kernel)

        return new_covariances

    def expand_full_kernel(self, covariance: Covariance) -> List[Covariance]:
        """Expand full kernel.

        :param covariance: Covariance to expand.
        :return:
        """
        result = self.expand_single_kernel(covariance)
        if covariance is None:
            pass
        elif not covariance.is_base():
            for i, operand in enumerate(covariance.raw_kernel.parts):
                covariance_operand = Covariance(operand)
                for e in self.expand_full_kernel(covariance_operand):
                    new_operands = covariance.raw_kernel.parts[:i] + [e.raw_kernel] \
                                   + covariance.raw_kernel.parts[i + 1:]
                    new_operands = [op.copy() for op in new_operands]
                    if covariance.is_prod():
                        prod_kern = new_operands[0]
                        for part in new_operands[1:]:
                            prod_kern *= part
                        result.append(Covariance(prod_kern))
                    elif covariance.is_sum():
                        prod_kern = new_operands[0]
                        for part in new_operands[1:]:
                            prod_kern += part
                        result.append(Covariance(prod_kern))
                    else:
                        raise TypeError(f'Unknown combination kernel class {covariance.__class__.__name__}')

        return result

    def expand_full_brute_force(self,
                                level: int,
                                max_n_models: int) -> List[Covariance]:
        """Enumerate all covariances in CKS grammar up to a depth with a size limit.

        :param level: Level of expansion.
        :param max_n_models: Maximum number of models in expansion.
        :return:
        """
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
                if number_of_models > max_n_models:
                    all_kernels += this_level
                    all_kernels = all_kernels[:max_n_models]
                    return all_kernels
            current_kernels = this_level
            all_kernels += this_level
            level -= 1

        return all_kernels


class BomsGrammar(CKSGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """
    _random_walk_geometric_dist_parameter: float
    _number_of_top_k_best: int
    _number_of_random_walks: int

    def __init__(self,
                 base_kernel_names: Optional[List[str]] = None,
                 hyperpriors: Optional[HyperpriorMap] = None):

        if hyperpriors is None:
            hyperpriors = boms_hyperpriors()

        super().__init__(base_kernel_names, hyperpriors)

        self._random_walk_geometric_dist_parameter = 1 / 3  # Termination probability.
        self._number_of_top_k_best = 3
        self._number_of_random_walks = 15

    def expand(self,
               seed_models: List[GPModel],
               verbose: int = 0) -> List[Covariance]:
        """Greedy and exploratory expansion of gp_models.

        :param seed_models: List of GP models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        # Exploration
        total_num_walks = self._number_of_random_walks
        candidates_random = self.expand_random(total_num_walks)

        # Exploitation
        evaluated_kernels = [kernel for kernel in seed_models if kernel.evaluated]
        fitness_score = [k.score for k in evaluated_kernels]
        candidates_best = self.expand_best(evaluated_kernels, fitness_score)

        # Concatenate
        candidates = candidates_best + candidates_random

        return candidates

    def expand_random(self, n_walks: int) -> List[Covariance]:
        """Geometric random walk covariances.

        :param n_walks: Total number of random walks.
        :return:
        """
        parameter = self._random_walk_geometric_dist_parameter
        depths = np.random.geometric(parameter, size=n_walks)
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

        :param selected_models:  List of GP models to expand.
        :param fitness_score: List of fitness scores for the selected models.
        :return:
        """
        new_kernels = []
        num_exploit_top = self._number_of_top_k_best
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

    def to_dict(self) -> dict:
        input_dict = super().to_dict()
        input_dict['random_walk_geometric_dist_parameter'] = self._random_walk_geometric_dist_parameter
        input_dict['number_of_top_k_best'] = self._number_of_top_k_best
        input_dict['number_of_random_walks'] = self._number_of_random_walks
        return input_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        random_walk_geometric_dist_parameter = input_dict.pop('random_walk_geometric_dist_parameter')
        number_of_top_k_best = input_dict.pop('number_of_top_k_best')
        number_of_random_walks = input_dict.pop('number_of_random_walks')

        grammar = super()._build_from_input_dict(input_dict)

        grammar._random_walk_geometric_dist_parameter = random_walk_geometric_dist_parameter
        grammar._number_of_top_k_best = number_of_top_k_best
        grammar._number_of_random_walks = number_of_random_walks

        return grammar


class RandomGrammar(BomsGrammar):
    """Random grammar randomly expands nodes using a CKS expansion"""

    def __init__(self,
                 base_kernel_names: List[str] = None,
                 hyperpriors: Optional[HyperpriorMap] = None):
        super().__init__(base_kernel_names, hyperpriors)

    def expand(self,
               seed_models: List[GPModel],
               verbose: int = 0) -> List[Covariance]:
        """Random expansion of nodes.

        :param seed_models:  List of GP models to expand.
        :param verbose: Verbosity mode.
        :return:
        """
        return self.expand_random(self._number_of_random_walks)

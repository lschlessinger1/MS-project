from autoks.kernel import get_all_1d_kernels
from evalg.selection import select_k_best


class BaseGrammar:

    def __init__(self, k):
        self.k = k

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize models

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self, models, model_scores, kernel_families):
        """ Get next round of candidate models from current models

        :param models:
        :param model_scores:
        :param kernel_families:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def select(self, models, model_scores):
        """ Select next round of models (default is top k models by objective)

        :param models:
        :param model_scores:
        :return:
        """
        return select_k_best(models, model_scores, self.k)


def sort_kernel_tree(kernel_tree):
    """ Sort kernel tree into canonical form

    :param kernel_tree:
    :return:
    """
    pass


def remove_duplicates(kernel_trees):
    """ Remove duplicate kernel trees (after sorting)

    :param kernel_trees:
    :return:
    """
    pass


class EvolutionaryGrammar(BaseGrammar):

    def __init__(self, k):
        super().__init__(k)

    def initialize(self, kernel_families, n_models, n_dims):
        """Naive initialization of all SE_i and RQ_i (for every dimension)

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        #
        kernels = get_all_1d_kernels(kernel_families, n_dims)

        # randomly initialize hyperparameters:
        for kernel in kernels:
            kernel.randomize()

        return kernels

    def expand(self, population, fitness_list, kernel_families):
        """ Perform crossover and mutation

        :param population: list of models
        :param fitness_list: list of model scores
        :param kernel_families: base kernels
        :return:
        """
        offspring = population.copy()
        return offspring


class BOMSGrammar(BaseGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """

    def __init__(self, k=600):
        super().__init__(k)

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize models according to number of dimensions

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        # {SE, RQ, LIN, PER} if dataset is 1D
        # {SE_i} + {RQ_i} otherwise
        kernels = []
        return kernels

    def expand(self, active_set, model_scores, kernel_families):
        """ Greedy and exploratory expansion of kernels

        :param active_set: list of models
        :param model_scores: list of
        :param kernel_families:
        :return:
        """
        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        pass

    def select(self, active_set, exp_imp_list):
        """ Select top 600 models according to expected improvement

        :param active_set:
        :param exp_imp_list:
        :return:
        """
        pass


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self, k):
        super().__init__(k)

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize with all base kernel families applied to all input dimensions

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        kernels = []
        return kernels

    def expand(self, models, model_scores, kernel_families):
        """ Greedy expansion of nodes

        :param models:
        :param model_scores:
        :param kernel_families:
        :return:
        """
        # choose highest scoring kernel (using BIC) and expand it by applying all possible operators
        # CFG:
        # 1) Any subexpression S can be replaced with S + B, where B is any base kernel family.
        # 2) Any subexpression S can be replaced with S x B, where B is any base kernel family.
        # 3) Any base kernel B may be replaced with any other base kernel family B'
        pass

    def select(self, active_set, model_scores):
        """ Select all

        :param active_set:
        :param model_scores:
        :return:
        """
        pass


class AdditiveGPGrammar(BaseGrammar):
    """
    Additive Gaussian Processes (Duvenaud et al., 2011)
    """

    def __init__(self, k):
        super().__init__(k)

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize the first order additive kernel (r = 1)

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        kernels = []
        return kernels

    def expand(self, models, fitness_list, kernel_families):
        """ Compute next order of interaction

        :param models:
        :param fitness_list:
        :param kernel_families:
        :return:
        """
        # for r=1...R (R <= D)
        # compute next order of  interaction for additive GPs
        # R is the maximum order of interaction (specify as parameter)
        pass

    def select(self, models, model_scores):
        """ Select all

        :param models:
        :param model_scores:
        :return:
        """
        pass

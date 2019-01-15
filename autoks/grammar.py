from GPy.kern import Kern, Prod, Add
from GPy.kern.src.kern import CombinationKernel

from autoks.kernel import get_all_1d_kernels, create_1d_kernel, get_kernel_mapping, kernel_to_infix, AKSKernel
from evalg.selection import select_k_best


class BaseGrammar:

    def __init__(self, n_parents):
        self.n_parents = n_parents

    def initialize(self, kernel_families, n_kernels, n_dims):
        """ Initialize kernels

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self, kernels, kernel_families, n_dims, verbose=False):
        """ Get next round of candidate kernels from current kernels

        :param kernels:
        :param kernel_families:
        :param n_dims: number of dimensions
        :param verbose:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def select_parents(self, kernels):
        """ Select next round of models (default is top k kernels by objective)

        :param kernels:
        :return:
        """
        return select_k_best(kernels, [k.score for k in kernels], self.n_parents)

    def select_offspring(self, kernels):
        """ Select next round of models (default is select all)

        :param kernels:
        :return:
        """
        return kernels


def argsort(unsorted_list):
    return [i[0] for i in sorted(enumerate(unsorted_list), key=lambda x: x[1])]


def sort_kernel(kernel):
    """ Sorts kernel tree
    """

    if not isinstance(kernel, CombinationKernel):
        return kernel
    elif isinstance(kernel, Kern):
        new_ops = []
        for op in kernel.parts:
            op_sorted = sort_kernel(op)
            if isinstance(op_sorted, CombinationKernel):
                new_ops.append(op_sorted)
            elif op_sorted is not None:
                new_ops.append(op_sorted)

        if len(new_ops) == 0:
            return None
        elif len(new_ops) == 1:
            return new_ops[0]
        else:
            k_sorted = sort_combination_kernel(kernel, new_ops)
            return k_sorted


def sort_combination_kernel(kernel, new_ops):
    """ Helper function to sort a combination kernel
    """
    # first sort by kernel name, then by active dim
    kmap = get_kernel_mapping()
    kmap_inv = {v: k for k, v in kmap.items()}
    # add sum and product entries
    kmap_inv[Prod] = 'PROD'
    kmap_inv[Add] = 'ADD'

    unsorted_kernel_names = []
    for operand in new_ops:
        if isinstance(operand, CombinationKernel):
            # to sort multiple combination kernels of the same type, use the parameter string
            param_str = ''.join(str(x) for x in operand.param_array)
            unsorted_kernel_names.append((kmap_inv[operand.__class__] + param_str))
        elif isinstance(operand, Kern):
            unsorted_kernel_names.append((kmap_inv[operand.__class__] + str(operand.active_dims[0])))
    ind = argsort(unsorted_kernel_names)
    sorted_ops = [new_ops[i] for i in ind]

    return kernel.__class__(sorted_ops)


def argunique(data):
    """ Get the indices of the unique elements in a list
    """
    unique_vals = set()
    unique_ind = []

    for i, datum in enumerate(data):
        if datum not in unique_vals:
            unique_vals.add(datum)
            unique_ind.append(i)

    return unique_ind


def remove_duplicates(data, values):
    """ Remove duplicates of data and replace with values
    """
    if len(data) != len(values):
        raise ValueError('The length of data must be be equal to the length of values')

    unique_ind = argunique(data)
    return [values[i] for i in unique_ind]


def remove_duplicate_kernels(kernels):
    """ Remove duplicate kernels
    """
    return remove_duplicates([kernel_to_infix(k) for k in kernels], kernels)


def remove_duplicate_aks_kernels(aks_kernels):
    """ Remove duplicate AKSKernel's
    """
    return remove_duplicates([kernel_to_infix(aks_kernel.kernel) for aks_kernel in aks_kernels], aks_kernels)


class EvolutionaryGrammar(BaseGrammar):

    def __init__(self, n_parents):
        super().__init__(n_parents)

    def initialize(self, kernel_families, n_kernels, n_dims):
        """Naive initialization of all SE_i and RQ_i (for every dimension)

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)

        # randomly initialize hyperparameters:
        for kernel in kernels:
            kernel.randomize()

        return kernels

    def expand(self, population, kernel_families, n_dims, verbose=False):
        """ Perform crossover and mutation

        :param population: list of models
        :param kernel_families: base kernels
        :param n_dims:
        :param verbose:
        :return:
        """
        offspring = population.copy()
        return offspring


class BOMSGrammar(BaseGrammar):
    """
    Bayesian optimization for automated model selection (Malkomes et al., 2016)
    """

    def __init__(self, n_parents=600):
        super().__init__(n_parents)

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize kernels according to number of dimensions

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        # {SE, RQ, LIN, PER} if dataset is 1D
        # {SE_i} + {RQ_i} otherwise
        kernels = []
        return kernels

    def expand(self, active_set, kernel_families, n_dims, verbose=False):
        """ Greedy and exploratory expansion of kernels

        :param active_set: list of kernels
        :param kernel_families:
        :param n_dims:
        :param verbose:
        :return:
        """
        # Exploit:
        # Add all neighbors (according to CKS grammar) of the best model seen thus far to active set
        # Explore:
        # Add 15 random walks (geometric dist w/ prob 1/3) from empty kernel to active set
        pass

    def select_offspring(self, active_set):
        """ Select top 600 kernels according to expected improvement

        :param active_set:
        :return:
        """
        pass


class CKSGrammar(BaseGrammar):
    """
    Structure Discovery in Nonparametric Regression through Compositional Kernel Search (Duvenaud et al., 2013)
    """

    def __init__(self, n_parents):
        super().__init__(n_parents)

    def initialize(self, kernel_families, n_kernels, n_dims):
        """ Initialize with all base kernel families applied to all input dimensions

        :param kernel_families:
        :param n_kernels:
        :param n_dims:
        :return:
        """
        kernels = get_all_1d_kernels(kernel_families, n_dims)
        kernels = [AKSKernel(kernel) for kernel in kernels]
        return kernels

    def expand(self, aks_kernels, kernel_families, n_dims, verbose=False):
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
            kernels_expanded = self.expand_full_kernel(kernel, ['+', '*'], n_dims, kernel_families)
            new_kernels += kernels_expanded

        new_kernels = remove_duplicate_kernels(new_kernels)
        new_kernels = [AKSKernel(kernel) for kernel in new_kernels]

        if verbose:
            print('Expanded kernels:')
            for k in new_kernels:
                k.pretty_print()

        return new_kernels

    @staticmethod
    def expand_single_kernel(kernel, ops, n_dims, base_kernels):
        is_kernel = isinstance(kernel, Kern)
        if not is_kernel:
            raise ValueError('Unknown kernel type %s' % kernel.__class__)

        kernels = []

        is_combo_kernel = isinstance(kernel, CombinationKernel)
        is_base_kernel = is_kernel and not is_combo_kernel

        for op in ops:
            for d in range(n_dims):
                for base_kernel_name in base_kernels:
                    if op == '+':
                        kernels.append(kernel + create_1d_kernel(base_kernel_name, d))
                    elif op == '*':
                        kernels.append(kernel * create_1d_kernel(base_kernel_name, d))
                    else:
                        raise ValueError('Unknown operation %s' % op)
        for base_kernel_name in base_kernels:
            if is_base_kernel:
                kernels.append(create_1d_kernel(base_kernel_name, kernel.active_dims[0]))
        return kernels

    @staticmethod
    def expand_full_kernel(kernel, operators, n_dims, base_kernels):
        result = CKSGrammar.expand_single_kernel(kernel, operators, n_dims, base_kernels)
        if kernel is None:
            pass
        elif isinstance(kernel, CombinationKernel):
            for i, operand in enumerate(kernel.parts):
                for e in CKSGrammar.expand_full_kernel(operand, operators, n_dims, base_kernels):
                    new_operands = kernel.parts[:i] + [e] + kernel.parts[i + 1:]
                    new_operands = [op.copy() for op in new_operands]
                    if isinstance(kernel, Prod):
                        result.append(Prod(new_operands))
                    elif isinstance(kernel, Add):
                        result.append(Add(new_operands))
                    else:
                        raise RuntimeError('Unknown combination kernel class:', kernel.__class__)
        elif isinstance(kernel, Kern):
            # base kernel
            pass
        else:
            raise ValueError('Unknown kernel class:', kernel.__class__)
        return result

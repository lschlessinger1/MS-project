import numpy as np
from GPy.kern import RBF, RatQuad, Linear, StdPeriodic, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern

from autoks.util import remove_duplicates, argsort


class AKSKernel:
    """AKS kernel wrapper
    """

    def __init__(self, kernel, scored=False, nan_scored=False):
        self.kernel = kernel
        self.scored = scored
        self.nan_scored = nan_scored
        self._score = None

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score
        # Update scored as well
        self.scored = True

    def pretty_print(self):
        print(str(self))

    def print_full(self):
        print(kernel_to_infix(self.kernel, show_params=True))

    def __str__(self):
        return kernel_to_infix(self.kernel)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={kernel_to_infix(self.kernel, show_params=True)!r}, score=' \
            f'{self.score!r}) '


def get_kernel_mapping():
    return dict(zip(get_allowable_kernels(), get_matching_kernels()))


def get_allowable_kernels():
    return ['SE', 'RQ', 'LIN', 'PER']


def get_matching_kernels():
    return [RBF, RatQuad, Linear, StdPeriodic]


def get_all_1d_kernels(base_kernels, n_dims):
    """

    :param base_kernels:
    :param n_dims: number of dimensions
    :return: list of models of size len(base_kernels) * n_dims
    """
    kernel_mapping = get_kernel_mapping()
    models = []

    for kern_fam in base_kernels:
        kern_map = kernel_mapping[kern_fam]
        for d in range(n_dims):
            kernel = create_1d_kernel(kern_fam, d, kernel_mapping=kernel_mapping, kernel_map=kern_map)
            models.append(kernel)

    return models


def create_1d_kernel(kernel_family, active_dim, kernel_mapping=None, kernel_map=None):
    if not kernel_mapping:
        kernel_mapping = get_kernel_mapping()
        if not kernel_map:
            kernel_map = kernel_mapping[kernel_family]
    return kernel_map(input_dim=1, active_dims=[active_dim])


def subkernel_expression(kernel, show_params=False):
    kernel_families = get_allowable_kernels()
    kernel_mapping = get_kernel_mapping()
    matching_base_kerns = [kern_fam for kern_fam in kernel_families if isinstance(kernel, kernel_mapping[kern_fam])]
    # assume only 1 active dimension
    dim = kernel.active_dims[0]
    # assume only 1 matching base kernel
    base_kernel = matching_base_kerns[0]

    kern_str = base_kernel + str(dim)
    if show_params:
        params = list(zip(kernel.parameter_names(), kernel.param_array))
        param_str = ', '.join(tuple([name + '=' + "{:.6f}".format(val) for name, val in params]))
        kern_str += '(' + param_str + ')'
    return kern_str


def in_order(root, tokens=[]):
    if root is not None:
        if isinstance(root, CombinationKernel):

            for child in root.parts:

                if isinstance(child, CombinationKernel):
                    if isinstance(child, Add):
                        op = '+'
                    elif isinstance(child, Prod):
                        op = '*'

                    children = in_order(child, tokens=[])
                    tokens += [children]

                elif isinstance(child, Kern):
                    tokens += [child]

            if isinstance(root, Add):
                op = '+'
            elif isinstance(root, Prod):
                op = '*'
            else:
                raise TypeError('Unrecognized operation %s' % root.__class__.__name__)

            tokens = join_operands(tokens, op)
        elif isinstance(root, Kern):
            tokens += [root]

    return tokens


def tokenize(list_nd):
    if not list_nd:
        return []
    if isinstance(list_nd, list):
        return ['('] + [tokenize(s) for s in list_nd] + [')']
    return list_nd


def flatten(list_nd):
    return [list_nd] if not isinstance(list_nd, list) else [x for X in list_nd for x in flatten(X)]


def remove_outer_parens(list_nd):
    if len(list_nd) >= 2:
        if list_nd[0] == '(' and list_nd[-1] == ')':
            return list_nd[1:-1]
        else:
            raise ValueError('List must start with \'(\' and end with \')\' ')
    else:
        raise ValueError('List must have length >= 2')


def join_operands(operands, operator):
    joined = []
    for i, operand in enumerate(operands):
        joined += [operand]
        if i < len(operands) - 1:
            joined += [operator]
    return joined


def kernel_to_infix_tokens(kernel):
    in_order_traversal = in_order(kernel, tokens=[])
    infix_tokens = flatten(tokenize(in_order_traversal))
    # for readability, remove outer parentheses
    infix_tokens = remove_outer_parens(infix_tokens)
    return infix_tokens


def tokens_to_str(tokens, show_params=False):
    token_string = ''
    for i, token in enumerate(tokens):
        if isinstance(token, Kern):
            token_string += subkernel_expression(token, show_params=show_params)
        else:
            token_string += token

        if i < len(tokens) - 1:
            token_string += ' '

    return token_string


def kernel_to_infix(kernel, show_params=False):
    return tokens_to_str(kernel_to_infix_tokens(kernel), show_params=show_params)


def apply_op(left, right, operator):
    if operator == '+':
        return left + right
    elif operator == '*':
        return left * right
    else:
        raise ValueError('Unknown operator %s' % operator)


def eval_binexp_tree(root):
    if root is not None:
        if isinstance(root.value, Kern):
            return root.value

        left_node = eval_binexp_tree(root.left)
        right_node = eval_binexp_tree(root.right)

        operator = root.value

        return apply_op(left_node, right_node, operator)


def tree_to_kernel(tree):
    return eval_binexp_tree(tree.root)


def n_base_kernels(kernel):
    """Count the number of base kernels."""
    count = [0]

    def count_base_kernels(kern):
        if isinstance(kern, Kern):
            if not isinstance(kern, CombinationKernel):
                count[0] += 1

    kernel.traverse(count_base_kernels)
    return count[0]


def covariance_distance(kernels, X):
    """Euclidean distance of all pairs kernels"""
    # For each pair of kernel matrices, compute Euclidean distance
    n_kernels = len(kernels)
    dists = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(i + 1, n_kernels):
            dists[i, j] = kernel_l2_dist(kernels[i], kernels[j], X)
    # Make symmetric
    dists = (dists + dists.T) / 2.
    return dists


def kernel_l2_dist(kernel_1, kernel_2, X):
    """Compute Euclidean distance between two kernel matrices."""
    dist = np.linalg.norm(kernel_1.K(X) - kernel_2.K(X))
    return dist


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


def remove_duplicate_kernels(kernels):
    """ Remove duplicate kernels
    """
    return remove_duplicates([kernel_to_infix(k) for k in kernels], kernels)


def remove_duplicate_aks_kernels(aks_kernels):
    """ Remove duplicate AKSKernel's
    """
    return remove_duplicates([kernel_to_infix(aks_kernel.kernel) for aks_kernel in aks_kernels], aks_kernels)


def additive_form(kernel):
    if isinstance(kernel, Prod):
        # Distribute kernel parts if necessary
        additive_ops = [additive_form(part) for part in kernel.parts]
        additive_kernel = additive_ops[0]

        for additive_op in additive_ops[1:]:
            if isinstance(additive_kernel, Add):
                additive_parts = [additive_form(op * additive_op) for op in additive_kernel.parts]
                additive_kernel = Add(additive_parts)
            elif isinstance(additive_op, Add):
                additive_parts = [additive_form(op * additive_kernel) for op in additive_op.parts]
                additive_kernel = Add(additive_parts)
            else:
                additive_kernel *= additive_op

        return sort_kernel(additive_kernel)
    elif isinstance(kernel, Add):
        # Make all kernel parts additive
        additive_parts = [additive_form(part) for part in kernel.parts]
        additive_kernel = Add(additive_parts)

        return sort_kernel(additive_kernel)
    elif isinstance(kernel, Kern):
        # Base kernel
        return kernel
    else:
        raise TypeError('%s is not a subclass of %s' % (kernel.__class__.__name__, Kern.__name__))


def additive_part_to_vec(additive_part, base_kernels, n_dims):
    # convert product into vector
    # ex: k1 * k1 * k2 * k4 ---> [2, 1, 0, 1]

    if isinstance(additive_part, Add):
        raise TypeError('additive_part cannot be a sum')

    all_kerns = get_all_1d_kernels(base_kernels, n_dims)
    vec_length = len(base_kernels) * n_dims
    vec = np.zeros(vec_length)

    if isinstance(additive_part, Prod):
        # has parts
        parts = additive_part.parts
    elif isinstance(additive_part, Kern):
        # Base kernel
        parts = [additive_part]
    else:
        raise TypeError('Unknown kernel %s' % additive_part.__class__.__name__)

    # count number of each kernel type in additive part
    for i, op in enumerate(parts):
        for j, kern in enumerate(all_kerns):
            same_kernel = isinstance(op, kern.__class__) and op.active_dims == kern.active_dims
            if same_kernel:
                vec[j] += 1

    return vec


def kernel_vec_avg_dist(kvecs1, kvecs2):
    """Average Euclidean distance between two lists of vectors."""

    total_dist = 0
    for kv1 in kvecs1:
        for kv2 in kvecs2:
            dist = np.linalg.norm(kv1 - kv2)
            total_dist += dist

    n = len(kvecs1) + len(kvecs2)
    avg_dist = total_dist / n

    return avg_dist


def all_pairs_avg_dist(kernels, base_kernels, n_dims):
    """ Calculates the mean distance between all pairs of kernels

    Can be thought of as a diversity score of a population of kernels
    """
    from scipy.special import comb
    # first change each kernel into additive form
    additive_kernels = [additive_form(kernel) for kernel in kernels]

    # For each kernel, convert each additive part into a vector
    kernel_vecs = []
    for additive_kernel in additive_kernels:
        kernel_vec = []
        if isinstance(additive_kernel, CombinationKernel):
            for additive_part in additive_kernel.parts:
                vec = additive_part_to_vec(additive_part, base_kernels, n_dims)
                kernel_vec.append(vec)
        elif isinstance(additive_kernel, Kern):
            # base kernel
            additive_part = additive_kernel
            vec = additive_part_to_vec(additive_part, base_kernels, n_dims)
            kernel_vec.append(vec)
        else:
            raise TypeError('Unknown kernel %s' % additive_kernel.__class__.__name__)

        kernel_vecs.append(kernel_vec)

    # compute average Eucldiean distance for all pairs of kernels
    all_pairs_total_dist = 0
    for i, v in enumerate(kernel_vecs):
        for u in kernel_vecs[i + 1:]:
            # compute dist between v and u
            avg_dist = kernel_vec_avg_dist(v, u)
            all_pairs_total_dist += avg_dist

    n_pairs = int(comb(N=len(kernel_vecs), k=2))
    all_pairs_avg_dist = all_pairs_total_dist / n_pairs
    return all_pairs_avg_dist

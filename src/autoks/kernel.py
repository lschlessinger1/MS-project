import warnings
from typing import List, Type, Union, Optional, Dict

import numpy as np
from GPy import Parameterized
from GPy.core.parameterization.priors import Prior
from GPy.kern import RBF, RatQuad, Linear, StdPeriodic, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern
from scipy.special import comb

from src.autoks.hyperprior import Hyperpriors, Hyperprior
from src.autoks.util import remove_duplicates, arg_sort, join_operands, tokenize, flatten, remove_outer_parens
from src.evalg.encoding import BinaryTree, BinaryTreeNode, infix_tokens_to_postfix_tokens, \
    postfix_tokens_to_binexp_tree


class KernelNode(BinaryTreeNode):
    _value: Union[Kern, str]

    def __init__(self,
                 value: Union[Kern, str],
                 parent: Optional[BinaryTreeNode] = None,
                 left: Optional[BinaryTreeNode] = None,
                 right: Optional[BinaryTreeNode] = None):
        super().__init__(value, parent, left, right)

    def _value_to_label(self, value: Union[Kern, str]) -> str:
        if isinstance(value, Kern):
            return subkernel_expression(value)
        else:
            return str(value)

    def _value_to_html(self, value) -> str:
        if isinstance(value, Kern):
            return subkernel_expression(value, html_like=True)
        else:
            return '<' + str(value) + '>'


class KernelTree(BinaryTree):
    _root: Optional[KernelNode]

    def __init__(self, root: Optional[KernelNode] = None):
        super().__init__(root)


class AKSKernel:
    """AKS kernel wrapper
    """
    kernel: Kern
    lik_params: Optional[np.ndarray]
    evaluated: bool
    nan_scored: bool
    expanded: bool

    def __init__(self, kernel, lik_params=None, evaluated=False, nan_scored=False, expanded=False):
        self.kernel = kernel
        self.lik_params = lik_params
        self.evaluated = evaluated
        self.nan_scored = nan_scored
        self.expanded = expanded
        self._score = None

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score
        # Update evaluated as well
        self.evaluated = True

    def to_binary_tree(self) -> KernelTree:
        """Get the binary tree representation of the kernel

        :return:
        """
        infix_tokens = kernel_to_infix_tokens(self.kernel)
        postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
        tree = postfix_tokens_to_binexp_tree(postfix_tokens, bin_tree_node_cls=KernelNode, bin_tree_cls=KernelTree)
        return tree

    def to_additive_form(self) -> None:
        """Convert the kernel to additive form.

        :return:
        """
        self.kernel = additive_form(self.kernel)

    def pretty_print(self) -> None:
        """Pretty print the kernel.

        :return:
        """
        print(str(self))

    def print_full(self) -> None:
        """Print the verbose version of the kernel.

        :return:
        """
        print(kernel_to_infix(self.kernel, show_params=True))

    def __str__(self):
        return kernel_to_infix(self.kernel)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={kernel_to_infix(self.kernel, show_params=True)!r}, ' \
            f'lik_params={self.lik_params!r}, evaluated={self.evaluated!r}, nan_scored={self.nan_scored!r}, ' \
            f'expanded={self.expanded!r}, score={self.score!r}) '


def pretty_print_aks_kernels(aks_kernels: List[AKSKernel],
                             kernel_type_label: Optional[str] = None):
    n_kernels = len(aks_kernels)

    plural_suffix = 's' if n_kernels > 1 else ''
    ending = f'kernel{plural_suffix}:'
    if kernel_type_label is not None:
        message = f'{n_kernels} {kernel_type_label} {ending}'
    else:
        message = f'{n_kernels} {ending}'
    message = message.capitalize()
    print(message)
    for k in aks_kernels:
        k.pretty_print()
    print('')


def get_kernel_mapping() -> Dict[str, Type[Kern]]:
    """Get the map from allowable kernels to the corresponding class.

    :return:
    """
    return {
        'SE': RBF,
        'RQ': RatQuad,
        'LIN': Linear,
        'PER': StdPeriodic
    }


def get_allowable_kernels() -> List[str]:
    """Get all allowable kernel families.

    :return:
    """
    return list(get_kernel_mapping().keys())


def get_matching_kernels() -> List[Type[Kern]]:
    """Get all allowable kernel family classes.

    :return:
    """
    return list(get_kernel_mapping().values())


def get_all_1d_kernels(base_kernels: List[str],
                       n_dims: int,
                       hyperpriors: Optional[Hyperpriors] = None) -> List[Kern]:
    """Get all possible 1-D kernels.

    :param base_kernels:
    :param n_dims: number of dimensions
    :param hyperpriors:
    :return: list of models of size len(base_kernels) * n_dims
    """
    kernel_mapping = get_kernel_mapping()
    models = []

    for kern_fam in base_kernels:
        kern_cls = kernel_mapping[kern_fam]
        for d in range(n_dims):
            hyperprior = None if hyperpriors is None else hyperpriors[kern_fam]
            kernel = create_1d_kernel(kern_fam, d, kernel_mapping=kernel_mapping, kernel_cls=kern_cls,
                                      hyperpriors=hyperprior)
            models.append(kernel)

    return models


def create_1d_kernel(kernel_family: str,
                     active_dim: int,
                     kernel_mapping: Optional[Dict[str, Type[Kern]]] = None,
                     kernel_cls: Optional[Type[Kern]] = None,
                     hyperpriors: Optional[Hyperprior] = None) -> Kern:
    """Create a 1D kernel.

    :param kernel_family:
    :param active_dim:
    :param kernel_mapping:
    :param kernel_cls:
    :param hyperpriors:
    :return:
    """
    if not kernel_mapping:
        kernel_mapping = get_kernel_mapping()
        if not kernel_cls:
            kernel_cls = kernel_mapping[kernel_family]

    kernel = kernel_cls(input_dim=1, active_dims=[active_dim])

    if hyperpriors is not None:
        kernel = set_priors(kernel, hyperpriors)

    return kernel


def set_priors(param: Parameterized, priors: Dict[str, Prior]) -> Parameterized:
    param_copy = param.copy()
    for param_name in priors:
        if param_name in param_copy.parameter_names():
            param_copy[param_name].set_prior(priors[param_name], warning=False)
        else:
            warnings.warn(f'parameter {param_name} not found in {param.__class__.__name__}.')
    return param_copy


def subkernel_expression(kernel: Kern,
                         show_params: bool = False,
                         html_like: bool = False) -> str:
    """Construct a subkernel expression

    :param kernel:
    :param show_params:
    :param html_like:
    :return:
    """
    kernel_families = get_allowable_kernels()
    kernel_mapping = get_kernel_mapping()
    matching_base_kerns = [kern_fam for kern_fam in kernel_families if isinstance(kernel, kernel_mapping[kern_fam])]
    # assume only 1 active dimension
    dim = kernel.active_dims[0]
    # assume only 1 matching base kernel
    base_kernel = matching_base_kerns[0]

    if not html_like:
        kern_str = base_kernel + str(dim)
    else:
        kern_str = '<%s<SUB><FONT POINT-SIZE="8">%s</FONT></SUB>>' % (base_kernel, dim)

    if show_params:
        params = list(zip(kernel.parameter_names(), kernel.param_array))
        param_str = ', '.join(tuple([name + '=' + "{:.6f}".format(val) for name, val in params]))
        kern_str += '(' + param_str + ')'
    return kern_str


def in_order(root: Kern,
             tokens: list = []) -> List[str]:
    """In-order traversal of a kernel tree.

    :param root:
    :param tokens:
    :return: The infix expression produced by the in-order traversal.
    """
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


def kernel_to_infix_tokens(kernel: Kern) -> List[str]:
    """Convert kernel to a list of infix tokens.

    :param kernel:
    :return:
    """
    in_order_traversal = in_order(kernel, tokens=[])
    infix_tokens = flatten(tokenize(in_order_traversal))
    # for readability, remove outer parentheses
    infix_tokens = remove_outer_parens(infix_tokens)
    return infix_tokens


def tokens_to_str(tokens: list,
                  show_params: bool = False) -> str:
    """Convert a list of kernel tokens to a string

    :param tokens:
    :param show_params:
    :return:
    """
    token_string = ''
    for i, token in enumerate(tokens):
        if isinstance(token, Kern):
            token_string += subkernel_expression(token, show_params=show_params)
        else:
            token_string += token

        if i < len(tokens) - 1:
            token_string += ' '

    return token_string


def kernel_to_infix(kernel: Kern,
                    show_params: bool = False) -> str:
    """Get the infix string of a kernel.

    :param kernel:
    :param show_params:
    :return:
    """
    return tokens_to_str(kernel_to_infix_tokens(kernel), show_params=show_params)


def apply_op(left: Kern,
             right: Kern,
             operator: str) -> Kern:
    """Apply binary operator to two kernels.

    :param left:
    :param right:
    :param operator:
    :return:
    """
    if operator == '+':
        return left + right
    elif operator == '*':
        return left * right
    else:
        raise ValueError(f'Unknown operator {operator}')


def eval_binexp_tree(root: BinaryTreeNode) -> Kern:
    """Evaluate a binary expression tree.

    :param root:
    :return:
    """
    if root is not None:
        if isinstance(root.value, Kern):
            return root.value

        left_node = eval_binexp_tree(root.left)
        right_node = eval_binexp_tree(root.right)

        operator = root.value

        return apply_op(left_node, right_node, operator)


def tree_to_kernel(tree: BinaryTree) -> Kern:
    """Convert a binary tree to a kernel.

    :param tree:
    :return:
    """
    return eval_binexp_tree(tree.root)


def n_base_kernels(kernel: Kern) -> int:
    """Count the number of base kernels.

    :param kernel:
    :return:
    """
    count = [0]

    def count_base_kernels(kern):
        if isinstance(kern, Kern):
            if not isinstance(kern, CombinationKernel):
                count[0] += 1

    kernel.traverse(count_base_kernels)
    return count[0]


def covariance_distance(kernels: List[Kern],
                        x: np.ndarray) -> np.ndarray:
    """Euclidean distance of all pairs kernels.

    :param kernels:
    :param x:
    :return:
    """
    # For each pair of kernel matrices, compute Euclidean distance
    n_kernels = len(kernels)
    dists = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(i + 1, n_kernels):
            dists[i, j] = kernel_l2_dist(kernels[i], kernels[j], x)
    # Make symmetric
    dists = (dists + dists.T) / 2.
    return dists


def kernel_l2_dist(kernel_1: Kern,
                   kernel_2: Kern,
                   x: np.ndarray) -> float:
    """Euclidean distance between two kernel matrices.

    :param kernel_1:
    :param kernel_2:
    :param x:
    :return:
    """
    return np.linalg.norm(kernel_1.K(x, x) - kernel_2.K(x, x))


def sort_kernel(kernel: Kern) -> Union[Kern, None]:
    """Sorts kernel tree.

    :param kernel:
    :return:
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


def sort_combination_kernel(kernel: Kern,
                            new_ops: List[Kern]) -> Kern:
    """Helper function to sort a combination kernel.

    :param kernel:
    :param new_ops:
    :return:
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
    ind = arg_sort(unsorted_kernel_names)
    sorted_ops = [new_ops[i] for i in ind]

    return kernel.__class__(sorted_ops)


def remove_duplicate_kernels(kernels: List[Kern]) -> List[Kern]:
    """Remove duplicate kernels.

    :param kernels:
    :return:
    """
    return remove_duplicates([kernel_to_infix(k) for k in kernels], kernels)


def remove_duplicate_aks_kernels(kernels: List[AKSKernel]) -> List[AKSKernel]:
    """Remove duplicate AKSKernel's.

    prioritizing when removing duplicates
        1. highest score
        2. not nan scored
        3. evaluated

    :param kernels:
    :return:
    """

    unevaluated_kernels = [kernel for kernel in kernels if not kernel.evaluated and not kernel.nan_scored]
    evaluated_kernels = [kernel for kernel in kernels if kernel.evaluated]
    nan_scored_kernels = [kernel for kernel in kernels if kernel.nan_scored]
    # Prioritize highest scoring kernel for duplicates
    sorted_evaluated_kernels = sorted(evaluated_kernels, key=lambda k: k.score, reverse=True)

    # Assume precedence by order.
    aks_kernels = sorted_evaluated_kernels + nan_scored_kernels + unevaluated_kernels
    return remove_duplicates([kernel_to_infix(aks_kernel.kernel) for aks_kernel in aks_kernels], aks_kernels)


def additive_form(kernel: Kern) -> Kern:
    """Get the additive form of a kernel.

    :param kernel:
    :return:
    """
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


def additive_part_to_vec(additive_part: Kern,
                         base_kernels: List[str],
                         n_dims: int) -> np.ndarray:
    """Get the vector encoding of an additive part.

    Convert product into vector
    ex: k1 * k1 * k2 * k4 ---> [2, 1, 0, 1]

    :param additive_part:
    :param base_kernels:
    :param n_dims:
    :return:
    """
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


def kernel_vec_avg_dist(kvecs1: List[np.ndarray],
                        kvecs2: List[np.ndarray]) -> float:
    """Average Euclidean distance between two lists of vectors.

    :param kvecs1:
    :param kvecs2:
    :return:
    """
    total_dist = 0
    for kv1 in kvecs1:
        for kv2 in kvecs2:
            dist = np.linalg.norm(kv1 - kv2)
            total_dist += dist

    n = len(kvecs1) + len(kvecs2)
    avg_dist = total_dist / n

    return avg_dist


def all_pairs_avg_dist(kernels: List[Kern],
                       base_kernels: List[str],
                       n_dims: int) -> float:
    """Mean distance between all pairs of kernels.

    Can be thought of as a diversity score of a population of kernels
    :param kernels:
    :param base_kernels:
    :param n_dims:
    :return:
    """
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

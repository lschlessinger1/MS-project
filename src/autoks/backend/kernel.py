import warnings
from typing import Type, Dict, List, Optional, Callable, Union

import numpy as np
from GPy import Parameterized
from GPy.core.parameterization.priors import Prior
from GPy.kern import Add, Prod
from GPy.kern import Kern, RationalQuadratic, RBF, LinScaleShift, StandardPeriodic
from GPy.kern.src.kern import CombinationKernel

from src.autoks.core.hyperprior import Hyperprior, Hyperpriors
from src.autoks.util import tokenize, flatten, remove_outer_parens, join_operands, arg_sort

RawKernelType = Kern

KERNEL_DICT = dict(
    SE=RBF,
    RQ=RationalQuadratic,
    LIN=LinScaleShift,
    PER=StandardPeriodic
)


def get_allowable_kernels() -> List[str]:
    """Get all allowable kernel families.

    :return:
    """
    return list(KERNEL_DICT.keys())


def get_matching_kernels() -> List[Type[RawKernelType]]:
    """Get all allowable kernel family classes.

    :return:
    """
    return list(KERNEL_DICT.values())


def create_1d_kernel(kernel_family: str,
                     active_dim: int,
                     kernel_mapping: Optional[Dict[str, Type[RawKernelType]]] = None,
                     kernel_cls: Optional[Type[RawKernelType]] = None,
                     hyperpriors: Optional[Hyperprior] = None) -> RawKernelType:
    """Create a 1D kernel.

    :param kernel_family:
    :param active_dim:
    :param kernel_mapping:
    :param kernel_cls:
    :param hyperpriors:
    :return:
    """
    if not kernel_mapping:
        kernel_mapping = KERNEL_DICT
        if not kernel_cls:
            kernel_cls = kernel_mapping[kernel_family]

    kernel = kernel_cls(input_dim=1, active_dims=[active_dim])

    if hyperpriors is not None:
        kernel = set_priors(kernel, hyperpriors)

    return kernel


def get_all_1d_kernels(base_kernel_names: List[str],
                       n_dims: int,
                       hyperpriors: Optional[Hyperpriors] = None) -> List[RawKernelType]:
    """Get all possible 1-D kernels.

    :param base_kernel_names:
    :param n_dims: number of dimensions
    :param hyperpriors:
    :return: list of kernels of size len(base_kernels) * n_dims
    """
    kernel_mapping = KERNEL_DICT
    kernels = []

    for kern_fam in base_kernel_names:
        kern_cls = kernel_mapping[kern_fam]
        for d in range(n_dims):
            hyperprior = None if hyperpriors is None else hyperpriors[kern_fam]
            kernel = create_1d_kernel(kern_fam, d, kernel_mapping=kernel_mapping, kernel_cls=kern_cls,
                                      hyperpriors=hyperprior)
            kernels.append(kernel)

    return kernels


def get_priors(kernel: RawKernelType):
    """Get the priors of a kernel (if they exists)"""
    # for now, make sure that priors are return in sorted order of parameters
    # and that all parameters have priors
    if kernel.priors.size != kernel.size:
        raise ValueError('All kernel parameters must have priors')

    priors = np.empty(kernel.priors.size, dtype=np.object)
    for prior, ind in kernel.priors.items():
        priors[ind] = prior
    return priors


def set_priors(param: Parameterized, priors: Dict[str, Prior]) -> Parameterized:
    param_copy = param.copy()
    for param_name in priors:
        if param_name in param_copy.parameter_names():
            param_copy[param_name].set_prior(priors[param_name], warning=False)
        else:
            warnings.warn(f'parameter {param_name} not found in {param.__class__.__name__}.')
    return param_copy


def encode_prior(prior: Prior) -> bytes:
    import pickle
    p_string = pickle.dumps(prior)
    return p_string


def decode_prior(prior_string: bytes) -> Prior:
    import pickle
    prior_new = pickle.loads(prior_string)
    return prior_new


def subkernel_expression(kernel: RawKernelType,
                         show_params: bool = False,
                         html_like: bool = False) -> str:
    """Construct a subkernel expression

    :param kernel:
    :param show_params:
    :param html_like:
    :return:
    """
    kernel_families = get_allowable_kernels()
    kernel_mapping = KERNEL_DICT
    matching_base_kerns = [kern_fam for kern_fam in kernel_families if isinstance(kernel, kernel_mapping[kern_fam])]
    # assume only 1 active dimension
    dim = kernel.active_dims[0] + 1
    # assume only 1 matching base kernel
    base_kernel = matching_base_kerns[0]

    if not html_like:
        kern_str = base_kernel + '_' + str(dim)
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


def kern_tokens_to_dict(tokens: List[Union[str, RawKernelType]]) -> List[Union[str, dict]]:
    new_tokens = []
    for token in tokens:
        new_token = token
        if isinstance(token, RawKernelType):
            new_token = token.to_dict()
        new_tokens.append(new_token)
    return new_tokens


def dict_to_kern(input_dict: dict) -> RawKernelType:
    return Kern.from_dict(input_dict)


def kernel_to_infix_tokens(kernel: RawKernelType) -> List[str]:
    """Convert kernel to a list of infix tokens.

    :param kernel:
    :return:
    """
    in_order_traversal = in_order(kernel, tokens=[])
    infix_tokens = flatten(tokenize(in_order_traversal))
    # for readability, remove outer parentheses
    if len(infix_tokens) > 1:
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
        if isinstance(token, RawKernelType):
            token_string += subkernel_expression(token, show_params=show_params)
        else:
            token_string += token

        if i < len(tokens) - 1:
            token_string += ' '

    return token_string


def kernel_to_infix(kernel: RawKernelType,
                    show_params: bool = False) -> str:
    """Get the infix string of a kernel.

    :param kernel:
    :param show_params:
    :return:
    """
    return tokens_to_str(kernel_to_infix_tokens(kernel), show_params=show_params)


def is_base_kernel(kernel: RawKernelType) -> bool:
    return isinstance(kernel, Kern) and not isinstance(kernel, CombinationKernel)


def is_sum_kernel(kernel: RawKernelType) -> bool:
    return isinstance(kernel, Add)


def is_prod_kernel(kernel: RawKernelType) -> bool:
    return isinstance(kernel, Prod)


def n_base_kernels(kernel: RawKernelType) -> int:
    """Count the number of base gp_models.

    :param kernel:
    :return:
    """
    return count_kernel_types(kernel, is_base_kernel)


def n_sum_kernels(kernel: RawKernelType) -> int:
    """Count the number of sum gp_models.

    :param kernel:
    :return:
    """
    return count_kernel_types(kernel, lambda k: isinstance(k, Add))


def n_prod_kernels(kernel: RawKernelType) -> int:
    """Count the number of product gp_models.

    :param kernel:
    :return:
    """
    return count_kernel_types(kernel, lambda k: isinstance(k, Prod))


def count_kernel_types(kernel: RawKernelType,
                       k_type_fn: Callable[[RawKernelType], bool]):
    """Count the number of gp_models of given type.

    :param kernel:
    :param k_type_fn:
    :return:
    """
    count = [0]

    def count_k_types(kern):
        if k_type_fn(kern):
            count[0] += 1

    kernel.traverse(count_k_types)
    return count[0]


def compute_kernel(kernel: RawKernelType,
                   x: np.ndarray,
                   x2: Optional[np.ndarray] = None) -> np.ndarray:
    if x2 is None:
        x2 = x
    return kernel.K(x, x2)


def sort_kernel(kernel: RawKernelType) -> Optional[RawKernelType]:
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


def sort_combination_kernel(kernel: RawKernelType,
                            new_ops: List[RawKernelType]) -> RawKernelType:
    """Helper function to sort a combination kernel.

    :param kernel:
    :param new_ops:
    :return:
    """
    # first sort by kernel name, then by active dim
    kmap = KERNEL_DICT
    kmap_inv = {v: k for k, v in kmap.items()}
    # add sum and product entries
    kmap_inv[Prod] = 'PROD'
    kmap_inv[Add] = 'ADD'

    unsorted_kernel_names = []
    for operand in new_ops:
        if isinstance(operand, CombinationKernel):
            # to sort multiple combination gp_models of the same type, use the parameter string
            param_str = ''.join(str(x) for x in operand.param_array)
            unsorted_kernel_names.append((kmap_inv[operand.__class__] + param_str))
        elif isinstance(operand, Kern):
            unsorted_kernel_names.append((kmap_inv[operand.__class__] + str(operand.active_dims[0])))
    ind = arg_sort(unsorted_kernel_names)
    sorted_ops = [new_ops[i] for i in ind]

    return kernel.__class__(sorted_ops)


def additive_form(kernel: RawKernelType) -> RawKernelType:
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


def kernels_to_kernel_vecs(kernels: List[RawKernelType],
                           base_kernels: List[str],
                           n_dims: int) -> List[np.ndarray]:
    # first change each kernel into additive form
    additive_kernels = [additive_form(kernel) for kernel in kernels]

    # For each kernel, convert each additive part into a vector
    kernel_vecs = []
    for additive_kernel in additive_kernels:
        if isinstance(additive_kernel, CombinationKernel):
            vecs = [additive_part_to_vec(additive_part, base_kernels, n_dims) for additive_part in
                    additive_kernel.parts]
            vec = np.vstack(vecs)
        elif isinstance(additive_kernel, Kern):
            # base kernel
            additive_part = additive_kernel
            vec = additive_part_to_vec(additive_part, base_kernels, n_dims)
            vec = np.atleast_2d(vec)
        else:
            raise TypeError('Unknown kernel %s' % additive_kernel.__class__.__name__)
        kernel_vecs.append(vec)
    return kernel_vecs


def additive_part_to_vec(additive_part: RawKernelType,
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
    elif n_sum_kernels(additive_part) > 0:
        raise TypeError('additive_part contains a sum kernel')

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


# Structural hamming distance functions

def decode_kernel(kern_dict_str: str) -> RawKernelType:
    """Convert kernel dictionary string to a kernel.

    :param kern_dict_str: A string of a Kern's dictionary representation.
    :return: A kernel from the kernel dictionary string
    """
    k_dict = eval(kern_dict_str)
    return Kern.from_dict(k_dict)


def encode_kernel(kern: RawKernelType) -> str:
    """Encode kernel.

    :param kern: The kernel to be encoded
    :return: The encoded kernel.
    """
    return str(kern.to_dict())

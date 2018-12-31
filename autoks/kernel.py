from GPy.kern import RBF, RatQuad, Linear, StdPeriodic, Add, Prod
from GPy.kern.src.kern import CombinationKernel, Kern


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
            kernel = kern_map(input_dim=1, active_dims=[d])
            models.append(kernel)

    return models


def subkernel_expression(kernel):
    kernel_families = get_allowable_kernels()
    kernel_mapping = get_kernel_mapping()
    matching_base_kerns = [kern_fam for kern_fam in kernel_families if isinstance(kernel, kernel_mapping[kern_fam])]
    # assume only 1 active dimension
    dim = kernel.active_dims[0]
    # assume only 1 matching base kernel
    base_kernel = matching_base_kerns[0]

    return base_kernel + str(dim)


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
                raise ValueError('Unrecognized operation %s' % root.__class__)

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


def tokens_to_str(tokens):
    token_string = ''
    for i, token in enumerate(tokens):
        if isinstance(token, Kern):
            token_string += subkernel_expression(token)
        else:
            token_string += token

        if i < len(tokens) - 1:
            token_string += ' '

    return token_string


def apply_op(left, right, operator):
    if operator == '+':
        return left + right
    elif operator == '*':
        return left * right
    else:
        print('Unknown operator %s' % operator)


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

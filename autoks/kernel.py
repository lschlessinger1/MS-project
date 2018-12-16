import GPy


def get_kernel_mapping():
    return dict(zip(get_allowable_kernels(), get_matching_kernels()))


def get_allowable_kernels():
    return ['SE', 'RQ', 'LIN', 'PER']


def get_matching_kernels():
    return [GPy.kern.RBF, GPy.kern.RatQuad, GPy.kern.Linear, GPy.kern.StdPeriodic]


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

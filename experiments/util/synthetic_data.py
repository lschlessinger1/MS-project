import GPy
import numpy as np


# the following were mostly taken or modified from:
# https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html


def sinosoid_1(n_samples: int = 50, n_dims: int = 1):
    # build a design matrix with a column of integers indicating the output
    X = np.random.rand(n_samples, n_dims) * 8

    # build a suitable set of observed variables
    y = np.sin(X) + np.random.randn(*X.shape) * 0.05
    y = np.sum(y[:, :, None], axis=1)
    return X, y


def sinosoid_2(n_samples: int = 30, n_dims: int = 1):
    # build a design matrix with a column of integers indicating the output
    X = np.random.rand(n_samples, n_dims) * 5

    # build a suitable set of observed variables
    y = np.sin(X) + np.random.randn(n_samples, n_dims) * 0.05 + 2.
    y = np.sum(y[:, :, None], axis=1)
    return X, y


def simple_periodic_1d(n_samples: int = 100):
    """1-D simple periodic data."""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_sin = np.sin(X * 1.5)
    noise = np.random.randn(*X.shape)
    y = (y_sin + noise).reshape(X.shape[0], 1)
    return X, y


def periodic_trend_1d(n_samples: int = 100):
    """1-D periodic trend"""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_sin = np.sin(X * 1.5)
    noise = np.random.randn(*X.shape)
    y = (X * (1 + y_sin) + noise * 2).reshape(X.shape[0], 1)
    return X, y


def linear_1d(n_samples: int = 100):
    """1-D linear data."""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    noise = np.random.randn(*X.shape)
    y = (X + noise).reshape(X.shape[0], 1)
    return X, y


def rbf_1d(n_samples: int = 100):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    f_true = np.random.multivariate_normal(np.zeros(n_samples), GPy.kern.RBF(1).K(X))
    y = np.array([np.random.poisson(np.exp(f)) for f in f_true])[:, None]
    return X, y


def cubic_sine_1d(n_samples: int = 151):
    X = (2 * np.pi) * np.random.random(n_samples) - np.pi
    y = np.sin(X) + np.random.normal(0, 0.2, n_samples)
    y = np.array([np.power(abs(y), float(1) / 3) * (1, -1)[y < 0] for y in y])
    X = X[:, None]
    y = y[:, None]
    return X, y


def toy_ARD_4d(n_samples: int = 300):
    # Create an artificial dataset where the values in the targets (Y)
    # only depend in dimensions 1 and 3 of the inputs (X). Run ARD to
    # see if this dependency can be recovered
    X1 = np.sin(np.sort(np.random.rand(n_samples, 1) * 10, 0))
    X2 = np.cos(np.sort(np.random.rand(n_samples, 1) * 10, 0))
    X3 = np.exp(np.sort(np.random.rand(n_samples, 1), 0))
    X4 = np.log(np.sort(np.random.rand(n_samples, 1), 0))
    X = np.hstack((X1, X2, X3, X4))

    y1 = np.asarray(2 * X[:, 0] + 3).reshape(-1, 1)
    y2 = np.asarray(4 * (X[:, 2] - 1.5 * X[:, 0])).reshape(-1, 1)
    y = np.hstack((y1, y2))

    y = np.dot(y, np.random.rand(2, 4))
    y = y + 0.2 * np.random.randn(y.shape[0], y.shape[1])
    y -= y.mean()
    y /= y.std()
    return X, y


def generate_data(n_samples: int = 100, n_dims: int = 1, min_terms: int = 2, max_terms: int = 10,
                  periodic: bool = False):
    """ Create regression problem

    :param n_samples:
    :param n_dims:
    :param min_terms:
    :param max_terms:
    :param periodic:
    :return:
    """
    import itertools

    def do_nothing(X):
        return X

    if periodic:
        pointwise_funcs = [np.cos, np.sin]
    else:
        pointwise_funcs = [np.cos, np.sin, np.exp, np.abs, do_nothing]

    operators = ['+', '*', '-', '/']

    n_operands = np.random.randint(min_terms, max_terms)
    n_operators = n_operands - 1 if n_operands > 1 else 0
    operator_sample = np.random.choice(operators, size=n_operators)
    pointwise_func_sample = np.random.choice(pointwise_funcs, size=n_operands)
    coeffs = np.random.random(n_operands)
    biases = np.random.random(n_operands)

    X = np.random.rand(n_samples, n_dims)

    f0 = pointwise_func_sample[0]
    c0 = coeffs[0]
    b0 = biases[0]
    y = c0 * f0(X) + b0
    y = np.sum(y[:, :, None], axis=1)
    for op, ufunc, c, b in itertools.zip_longest(operator_sample, pointwise_func_sample[1:],
                                                 coeffs[1:], biases[1:]):
        f_val = c * ufunc(X) + b
        f_val = np.sum(f_val[:, :, None])
        if op == '+':
            y += f_val
        elif op == '-':
            y -= f_val
        elif op == '*':
            y *= f_val
        elif op == '/':
            y /= f_val

    # pretty print the generated function:
    func_names = []
    for func in pointwise_func_sample:
        if func != do_nothing:
            func_names.append(func.__name__ + '(X)')
        else:
            func_names.append('X')
    pretty_str = ''
    for f, op in itertools.zip_longest(func_names, operator_sample):
        if op is not None:
            pretty_str += f + ' ' + op + ' '
        else:
            pretty_str += f
    print('y = f(X) =', pretty_str)

    return X, y

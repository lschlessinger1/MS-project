import GPy
import numpy as np

# the following were mostly taken or modified from:
# https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html
from src.experiments.util.data_util import SyntheticDataset, Input1DSyntheticDataset


class Sinosoid1Dataset(SyntheticDataset):

    def __init__(self, n_samples=50, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        # build a design matrix with a column of integers indicating the output
        x = np.random.rand(self.n_samples, self.input_dim) * 8

        # build a suitable set of observed variables
        y = np.sin(x) + np.random.randn(*x.shape) * 0.05
        y = np.sum(y[:, :, None], axis=1)
        return x, y


class Sinosoid2Dataset(SyntheticDataset):

    def __init__(self, n_samples=30, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        # build a design matrix with a column of integers indicating the output
        x = np.random.rand(self.n_samples, self.input_dim) * 5

        # build a suitable set of observed variables
        y = np.sin(x) + np.random.randn(self.n_samples, self.input_dim) * 0.05 + 2.
        y = np.sum(y[:, :, None], axis=1)
        return x, y


class SimplePeriodic1dDataset(Input1DSyntheticDataset):

    def __init__(self, n_samples=100, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        """1-D simple periodic data."""
        x = np.linspace(0, 10, self.n_samples).reshape(-1, 1)
        y_sin = np.sin(x * 1.5)
        noise = np.random.randn(*x.shape)
        y = (y_sin + noise).reshape(x.shape[0], 1)
        return x, y


class PeriodicTrend1dDataset(Input1DSyntheticDataset):

    def __init__(self, n_samples=100, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        """1-D periodic trend"""
        x = np.linspace(0, 10, self.n_samples).reshape(-1, 1)
        y_sin = np.sin(x * 1.5)
        noise = np.random.randn(*x.shape)
        y = (x * (1 + y_sin) + noise * 2).reshape(x.shape[0], 1)
        return x, y


class LinearTrend1dDataset(Input1DSyntheticDataset):

    def __init__(self, n_samples=100, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        """1-D linear data."""
        x = np.linspace(0, 10, self.n_samples).reshape(-1, 1)
        noise = np.random.randn(*x.shape)
        y = (x + noise).reshape(x.shape[0], 1)
        return x, y


class RBF1dDataset(Input1DSyntheticDataset):

    def __init__(self, n_samples=100, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        x = np.linspace(0, 10, self.n_samples).reshape(-1, 1)
        f_true = np.random.multivariate_normal(np.zeros(self.n_samples), GPy.kern.RBF(1).K(x))
        y = np.array([np.random.poisson(np.exp(f)) for f in f_true])[:, None]
        return x, y


class CubicSine1dDataset(Input1DSyntheticDataset):

    def __init__(self, n_samples=151, input_dim=1):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        x = (2 * np.pi) * np.random.random(self.n_samples) - np.pi
        y = np.sin(x) + np.random.normal(0, 0.2, self.n_samples)
        y = np.array([np.power(abs(y), float(1) / 3) * (1, -1)[y < 0] for y in y])
        x = x[:, None]
        y = y[:, None]
        return x, y


class ToyARD4dDataset(SyntheticDataset):

    def __init__(self, n_samples=300, input_dim=4):
        super().__init__(n_samples, input_dim)

    def load_or_generate_data(self):
        # Create an artificial dataset where the values in the targets (Y)
        # only depend in dimensions 1 and 3 of the inputs (x). Run ARD to
        # see if this dependency can be recovered
        x1 = np.sin(np.sort(np.random.rand(self.n_samples, 1) * 10, 0))
        x2 = np.cos(np.sort(np.random.rand(self.n_samples, 1) * 10, 0))
        x3 = np.exp(np.sort(np.random.rand(self.n_samples, 1), 0))
        x4 = np.log(np.sort(np.random.rand(self.n_samples, 1), 0))
        x = np.hstack((x1, x2, x3, x4))

        y1 = np.asarray(2 * x[:, 0] + 3).reshape(-1, 1)
        y2 = np.asarray(4 * (x[:, 2] - 1.5 * x[:, 0])).reshape(-1, 1)
        y = np.hstack((y1, y2))

        y = np.dot(y, np.random.rand(2, 4))
        y = y + 0.2 * np.random.randn(y.shape[0], y.shape[1])
        y -= y.mean()
        y /= y.std()
        return x, y


class SyntheticRegressionDataset(SyntheticDataset):
    min_terms: int
    max_terms: int
    periodic: bool

    def __init__(self, n_samples=100, input_dim=1, min_terms=2, max_terms=10, periodic=False):
        super().__init__(n_samples, input_dim)
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.periodic = periodic

    def load_or_generate_data(self):
        """Create regression problem."""
        import itertools

        def do_nothing(argv):
            return argv

        if self.periodic:
            pointwise_funcs = [np.cos, np.sin]
        else:
            pointwise_funcs = [np.cos, np.sin, np.exp, np.abs, do_nothing]

        operators = ['+', '*', '-', '/']

        n_operands = np.random.randint(self.min_terms, self.max_terms)
        n_operators = n_operands - 1 if n_operands > 1 else 0
        operator_sample = np.random.choice(operators, size=n_operators)
        pointwise_func_sample = np.random.choice(pointwise_funcs, size=n_operands)
        coeffs = np.random.random(n_operands)
        biases = np.random.random(n_operands)

        x = np.random.rand(self.n_samples, self.input_dim)

        f0 = pointwise_func_sample[0]
        c0 = coeffs[0]
        b0 = biases[0]
        y = c0 * f0(x) + b0
        y = np.sum(y[:, :, None], axis=1)
        for op, u_func, c, b in itertools.zip_longest(operator_sample, pointwise_func_sample[1:],
                                                      coeffs[1:], biases[1:]):
            f_val = c * u_func(x) + b
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
                func_names.append(func.__name__ + '(x)')
            else:
                func_names.append('x')
        pretty_str = ''
        for f, op in itertools.zip_longest(func_names, operator_sample):
            if op is not None:
                pretty_str += f + ' ' + op + ' '
            else:
                pretty_str += f
        print('y = f(x) =', pretty_str)

        return x, y


class BraninGenerator(SyntheticDataset):

    def __init__(self, n_samples=25, input_dim=2):
        super().__init__(n_samples, input_dim)

    @staticmethod
    def branin(x: np.ndarray) -> np.ndarray:
        """Branin function"""
        y = (x[:, 1] - 5.1 / (4 * np.pi ** 2) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10
        return y

    def load_or_generate_data(self):
        """
        2-dimensional Branin function defined over [-5, 10] x [0, 15]
        and a set of 25 observations.
        """

        # Training set defined as a 5 x 5 square:
        xg1 = np.linspace(-5, 10, 5)
        xg2 = np.linspace(0, 15, 5)
        x = np.zeros((xg1.size * xg2.size, 2))
        for i, x1 in enumerate(xg1):
            for j, x2 in enumerate(xg2):
                x[i + xg1.size * j, :] = [x1, x2]

        y = self.branin(x)[:, None]
        return x, y

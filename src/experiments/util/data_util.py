import os
from abc import ABC

import numpy as np
from GPy.kern import RBF, RatQuad, StdPeriodic, Linear
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment


def gen_dataset_paths(data_dir: str, file_suffix: str = '.csv'):
    """Return a list of dataset file paths.

    Assume that all data files are CSVs
    """
    file_paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_suffix):
                file_paths.append(os.path.join(root, file))

    return file_paths


def run_experiments(ds_generators, grammar, objective, base_kernels=None, **kwargs):
    for generator in ds_generators:
        print(f'Performing experiment on {generator.path}')
        X, y = generator.gen_dataset()

        if base_kernels is None:
            if X.shape[1] > 1:
                base_kernels = ['SE', 'RQ']
            else:
                base_kernels = ['SE', 'RQ', 'LIN', 'PER']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, **kwargs)
        experiment.run(title='Random Experiment')


def sample_gp(kernel, n_pts=500, noise_var=1):
    """Sample paths from a GP"""
    X = np.random.uniform(0., 1., (n_pts, kernel.input_dim))

    # zero-mean
    prior_mean = np.zeros(n_pts)
    prior_cov = kernel.K(X)

    # Generate a sample path
    Z = np.random.multivariate_normal(prior_mean, prior_cov)

    # additive Gaussian noise
    noise = np.random.randn(n_pts, 1) * np.sqrt(noise_var)
    y = Z.reshape(-1, 1) + noise

    return X, y


def cks_known_kernels():
    """Duvenaud, et al., 2013 Table 1"""
    se1 = RBF(1, active_dims=[0])
    se2 = RBF(1, active_dims=[1])
    se3 = RBF(1, active_dims=[2])
    se4 = RBF(1, active_dims=[3])
    rq1 = RatQuad(1, active_dims=[0])
    rq2 = RatQuad(1, active_dims=[1])
    per1 = StdPeriodic(1, active_dims=[0])
    lin1 = Linear(1, active_dims=[0])

    true_kernels = [se1 + rq1, lin1 * per1, se1 + rq2, se1 + se2 * per1 + se3,
                    se1 * se2, se1 * se2 + se2 * se3, (se1 + se2) * (se3 + se4)]
    return true_kernels

class DatasetGenerator:

    def gen_dataset(self):
        raise NotImplementedError('Must be implemented in a child class')


class SyntheticDatasetGenerator(DatasetGenerator, ABC):

    def __init__(self, n_samples, input_dim):
        self.n_samples = n_samples
        self.input_dim = input_dim


class Input1DSynthGenerator(SyntheticDatasetGenerator, ABC):

    def __init__(self, n_samples, input_dim):
        super().__init__(n_samples, input_dim)
        if self.input_dim != 1:
            raise ValueError('Input dimension must be 1')


class FileDatasetGenerator(DatasetGenerator):

    def __init__(self, file_path):
        # assume file type of CSV
        self.path = file_path

    def gen_dataset(self):
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        X, y = data[:, :-1], data[:, -1]
        return X, y


class KnownGPGenerator(DatasetGenerator):

    def __init__(self, kernel, noise_var, n_pts=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.n_pts = n_pts

    def gen_dataset(self):
        X, y = sample_gp(self.kernel, self.n_pts, self.noise_var)
        return X, y

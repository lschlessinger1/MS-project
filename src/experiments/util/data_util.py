import os
from abc import ABC
from typing import Iterable, Callable, List, Optional, Tuple

import numpy as np
from GPy.kern import RBF, RatQuad, StdPeriodic, Linear, Kern
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import CKSGrammar, BaseGrammar


class DatasetGenerator:

    def gen_dataset(self):
        raise NotImplementedError('Must be implemented in a child class')


class SyntheticDatasetGenerator(DatasetGenerator, ABC):
    n_samples: int
    input_dim: int

    def __init__(self, n_samples, input_dim):
        self.n_samples = n_samples
        self.input_dim = input_dim


class Input1DSynthGenerator(SyntheticDatasetGenerator, ABC):

    def __init__(self, n_samples, input_dim):
        super().__init__(n_samples, input_dim)
        if self.input_dim != 1:
            raise ValueError('Input dimension must be 1')


class FileDatasetGenerator(DatasetGenerator):
    file_path: str

    def __init__(self, file_path):
        # assume file type of CSV
        self.path = file_path

    def gen_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        X, y = data[:, :-1], data[:, -1]
        return X, y


class KnownGPGenerator(DatasetGenerator):
    kernel: Kern
    noise_var: float
    n_pts: int

    def __init__(self, kernel, noise_var, n_pts=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.n_pts = n_pts

    def gen_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = sample_gp(self.kernel, self.n_pts, self.noise_var)
        return X, y


def gen_dataset_paths(data_dir: str, file_suffix: str = '.csv') -> List[str]:
    """Return a list of dataset file paths.

    Assume that all data files are CSVs
    """
    file_paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_suffix):
                file_paths.append(os.path.join(root, file))

    return file_paths


def run_experiments(ds_generators: Iterable[DatasetGenerator], grammar: BaseGrammar, objective: Callable,
                    base_kernels: Optional[List[str]] = None, **kwargs) -> None:
    for generator in ds_generators:
        print(f'Performing experiment on {generator.path}')
        X, y = generator.gen_dataset()

        if base_kernels is None:
            base_kernels = CKSGrammar.get_base_kernels(X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, **kwargs)
        experiment.run(title='Random Experiment')


def sample_gp(kernel: Kern, n_pts: int = 500, noise_var: float = 1) -> Tuple[np.ndarray, np.ndarray]:
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


def cks_known_kernels() -> List[Kern]:
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

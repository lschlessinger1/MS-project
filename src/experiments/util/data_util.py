import os
from abc import ABC
from typing import Iterable, Callable, List, Optional, Tuple

import numpy as np
from GPy.kern import Kern
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment
from src.autoks.grammar import CKSGrammar, BaseGrammar
from src.autoks.kernel import kernel_to_infix, create_1d_kernel
from src.autoks.kernel_selection import KernelSelector


class Dataset:
    """Simple abstract class for datasets."""

    def load_or_generate_data(self):
        raise NotImplementedError('Must be implemented in a child class')

    def __repr__(self):
        return f'{self.__class__.__name__}'


class SyntheticDataset(Dataset, ABC):
    n_samples: int
    input_dim: int

    def __init__(self, n_samples, input_dim):
        self.n_samples = n_samples
        self.input_dim = input_dim

    def __repr__(self):
        return f'{self.__class__.__name__}('f' n={self.n_samples!r}, d={self.input_dim!r})'


class Input1DSyntheticDataset(SyntheticDataset, ABC):

    def __init__(self, n_samples, input_dim=1):
        super().__init__(n_samples, input_dim)
        if self.input_dim != 1:
            raise ValueError('Input dimension must be 1')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n={self.n_samples!r}, d={self.input_dim!r})'


class FileDataset(Dataset):
    file_path: str

    def __init__(self, file_path):
        # assume file type of CSV
        self.path = file_path

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        x, y = data[:, :-1], data[:, -1]
        return x, y

    def __repr__(self):
        return f'{self.__class__.__name__}('f' file_path={self.file_path!r}) '


class KnownGPDataset(Dataset):
    kernel: Kern
    noise_var: float
    n_pts: int

    def __init__(self, kernel, noise_var, n_pts=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.n_pts = n_pts

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = sample_gp(self.kernel, self.n_pts, self.noise_var)
        return x, y

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={kernel_to_infix(self.kernel, show_params=True)!r}, n=' \
            f'{self.n_pts!r}, noise={self.noise_var!r})'


def gen_dataset_paths(data_dir: str,
                      file_suffix: str = '.csv') -> List[str]:
    """Return a list of dataset file paths.

    Assume that all data files are CSVs
    """
    file_paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_suffix):
                file_paths.append(os.path.join(root, file))

    return file_paths


def run_experiments(datasets: Iterable[Dataset],
                    grammar: BaseGrammar,
                    kernel_selector: KernelSelector,
                    objective: Callable,
                    base_kernels: Optional[List[str]] = None,
                    **kwargs) -> None:
    for dataset in datasets:
        print(f'Performing experiment on \n {dataset}')
        x, y = dataset.load_or_generate_data()

        if base_kernels is None:
            base_kernels = CKSGrammar.get_base_kernels(x.shape[1])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        experiment = Experiment(grammar, kernel_selector, objective, base_kernels, x_train, y_train, x_test, y_test,
                                **kwargs)
        experiment.run(title='Random Experiment')


def sample_gp(kernel: Kern,
              n_pts: int = 500,
              noise_var: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Sample paths from a GP.

    :param kernel: the kernel from which to draw samples
    :param n_pts: number of data points
    :param noise_var: variance of additive gaussian noise
    :return:
    """
    x = np.random.uniform(0., 1., (n_pts, kernel.input_dim))

    # zero-mean
    prior_mean = np.zeros(n_pts)
    prior_cov = kernel.K(x, x)

    # Generate a sample path
    z = np.random.multivariate_normal(prior_mean, prior_cov)

    # additive Gaussian noise
    noise = np.random.randn(n_pts, 1) * np.sqrt(noise_var)
    y = z.reshape(-1, 1) + noise

    return x, y


def cks_known_kernels() -> List[Kern]:
    """Duvenaud, et al., 2013 Table 1"""
    se1 = create_1d_kernel('SE', 0)
    se2 = create_1d_kernel('SE', 1)
    se3 = create_1d_kernel('SE', 2)
    se4 = create_1d_kernel('SE', 3)
    rq1 = create_1d_kernel('RQ', 0)
    rq2 = create_1d_kernel('RQ', 1)
    per1 = create_1d_kernel('PER', 0)
    lin1 = create_1d_kernel('LIN', 0)

    true_kernels = [se1 + rq1, lin1 * per1, se1 + rq2, se1 + se2 * per1 + se3,
                    se1 * se2, se1 * se2 + se2 * se3, (se1 + se2) * (se3 + se4)]
    return true_kernels

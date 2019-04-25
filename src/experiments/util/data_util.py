from abc import ABC
from typing import List, Tuple

import numpy as np
from GPy.kern import Kern
from sklearn.model_selection import train_test_split

from src.autoks.backend.kernel import create_1d_kernel
from src.autoks.core.covariance import Covariance


class Dataset:
    """Simple abstract class for datasets."""

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError('Must be implemented in a child class')

    def split_train_test(self,
                         test_size=0.2,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y = self.load_or_generate_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, **kwargs)
        return x_train, x_test, y_train, y_test

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
        return f'{self.__class__.__name__}('f' file_path={self.path!r}) '


class KnownGPDataset(Dataset):
    kernel: Covariance
    noise_var: float
    n_pts: int

    def __init__(self, kernel, noise_var, n_pts=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.n_pts = n_pts

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = sample_gp(self.kernel.raw_kernel, self.n_pts, self.noise_var)
        return x, y

    def snr(self):
        # assume signal variance always = 1
        signal_variance = 1
        return signal_variance / self.noise_var

    def __repr__(self):
        return f'{self.__class__.__name__}('f'kernel={self.kernel.infix_full!r}, n=' \
            f'{self.n_pts!r}, SNR={self.snr() !r})'


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
    f_true = np.random.multivariate_normal(prior_mean, prior_cov)
    f_true = f_true[:, None]

    # additive Gaussian noise
    gaussian_noise_mean = 0
    gaussian_noise_std = np.sqrt(noise_var)
    gaussian_noise = np.random.normal(gaussian_noise_mean, gaussian_noise_std, n_pts)
    y = f_true + gaussian_noise

    return x, y


def cks_known_kernels() -> List[Kern]:
    """Duvenaud, et al., 2013 (Table 1)"""
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

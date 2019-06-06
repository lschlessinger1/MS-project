from typing import Tuple, List

import numpy as np
from GPy.kern import Kern

from src.autoks.backend.kernel import create_1d_kernel
from src.autoks.core.covariance import Covariance
from src.datasets.dataset import Dataset


class KnownGPDataset(Dataset):
    kernel: Covariance
    noise_var: float
    n_pts: int

    def __init__(self, kernel, noise_var, n_pts=100):
        super().__init__()
        self.kernel = kernel
        self.noise_var = noise_var
        self.n_pts = n_pts

    def load_or_generate_data(self) -> None:
        self.x, self.y = sample_gp(self.kernel.raw_kernel, self.n_pts, self.noise_var)

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
    gaussian_noise = np.random.normal(gaussian_noise_mean, gaussian_noise_std, (n_pts, 1))
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

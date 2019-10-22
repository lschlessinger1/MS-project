import numpy as np

from src.autoks.acquisition.util import get_quantiles
from src.autoks.gp_regression_models import KernelKernelGPModel


def expected_improvement(x: np.ndarray,
                         model: KernelKernelGPModel,
                         jitter: float = 0.01):
    """
    Computes the Expected Improvement per unit of cost.
    """
    m, s = model.predict(x)
    f_max = model.get_f_max()
    phi, Phi, u = get_quantiles(jitter, f_max, m, s)
    f_acqu = s * (u * Phi + phi)
    return f_acqu

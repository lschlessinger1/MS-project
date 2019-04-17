import numpy as np


# Adapted from Malkomes, 2016
# Bayesian optimization for automated model selection (BOMS)
# https://github.com/gustavomalkomes/automated_model_selection


def fix_numerical_problem(k: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """

    :param k:
    :param tolerance:
    :return:
    """
    d, v = np.linalg.eig(k)
    new_diagonal = d
    new_diagonal[new_diagonal < tolerance] = tolerance
    new_diagonal = np.diag(new_diagonal)
    k = v @ new_diagonal @ v.T
    k = (k + k.T) / 2
    chol_k = np.linalg.cholesky(k).T
    return chol_k

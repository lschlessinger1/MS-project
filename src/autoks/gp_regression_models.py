from typing import Optional

import numpy as np
from GPy.models import GPRegression

from src.autoks.backend.kernel import set_priors
from src.autoks.core.covariance import Covariance
from src.autoks.core.hyperprior import HyperpriorMap


class KernelKernelGPModel:

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 kernel_kernel: Covariance,
                 noise_var: Optional[float] = None,
                 exact_f_eval: bool = False,
                 optimizer: Optional[str] = 'lbfgsb',
                 max_iters: int = 1000,
                 optimize_restarts: int = 5,
                 verbose: bool = True,
                 kernel_kernel_hyperpriors: Optional[HyperpriorMap] = None):
        """

        :param x: 2d array of indices of distance builder
        :param y: model fitness scores
        :param kernel_kernel:
        :param noise_var:
        :param exact_f_eval:
        :param optimizer:
        :param max_iters:
        :param optimize_restarts:
        :param verbose:
        :param kernel_kernel_hyperpriors:
        """
        self.noise_var = noise_var
        self.exact_f_eval = exact_f_eval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        # Make sure input data consists only of positive integers.
        assert np.issubdtype(x.dtype, np.integer) and x.min() >= 0

        noise_var = y.var() * 0.01 if self.noise_var is None else self.noise_var
        normalize = x.size > 1  # only normalize if more than 1 observation.
        self._model = GPRegression(x, y, kernel_kernel.raw_kernel, noise_var=noise_var, normalizer=normalize)

        if kernel_kernel_hyperpriors is not None:
            if 'GP' in kernel_kernel_hyperpriors:
                # Set likelihood hyperpriors.
                likelihood_hyperprior = kernel_kernel_hyperpriors['GP']
                set_priors(self.model.likelihood, likelihood_hyperprior, in_place=True)
            if 'SE' in kernel_kernel_hyperpriors:
                # Set kernel hyperpriors.
                se_hyperprior = kernel_kernel_hyperpriors['SE']
                set_priors(self.model.kern, se_hyperprior, in_place=True)

        # Restrict variance if exact evaluations of the objective.
        if self.exact_f_eval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            if self.model.priors.size > 0:
                # FIXME: shouldn't need this case, but GPy doesn't have log Jacobian implemented for Logistic
                self.model.Gaussian_noise.constrain_positive(warning=False)
            else:
                self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

    def train(self):
        if self.max_iters > 0:
            # Update the model maximizing the marginal likelihood.
            if self.optimize_restarts == 1:
                self.model.optimize(optimizer=self.optimizer, max_iters=self.max_iters, messages=False,
                                    ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                             max_iters=self.max_iters, ipython_notebook=False, verbose=self.verbose,
                                             robust=True, messages=False)

    def _predict(self,
                 x: np.ndarray,
                 full_cov: bool,
                 include_likelihood: bool):
        if x.ndim == 1:
            x = x[None, :]
        m, v = self.model.predict(x, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self,
                x: np.ndarray,
                with_noise: bool = True):
        m, v = self._predict(x, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    @property
    def model(self):
        return self._model

    def get_f_max(self):
        """
        Returns the location where the posterior mean is takes its maximal value.
        """
        return self.model.predict(self.model.X)[0].max()

    def plot(self):
        import matplotlib.pyplot as plt
        self.model.plot(plot_limits=(0, self.model.kern.n_models - 1))
        plt.show()

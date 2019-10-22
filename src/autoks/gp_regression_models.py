from typing import Optional

import numpy as np
from GPy.models import GPRegression

from src.autoks.backend.kernel import set_priors
from src.autoks.core.covariance import Covariance
from src.autoks.core.hyperprior import HyperpriorMap


class KernelKernelGPModel:

    def __init__(self,
                 kernel_kernel: Optional[Covariance] = None,
                 noise_var: Optional[float] = None,
                 exact_f_eval: bool = False,
                 optimizer: Optional[str] = 'lbfgsb',
                 max_iters: int = 1000,
                 optimize_restarts: int = 5,
                 verbose: bool = True,
                 kernel_kernel_hyperpriors: Optional[HyperpriorMap] = None):
        """

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
        self.covariance = kernel_kernel
        self.kernel_hyperpriors = kernel_kernel_hyperpriors
        self.model = None

    def train(self):
        """Train (optimize) the model."""
        if self.max_iters > 0:
            # Update the model maximizing the marginal likelihood.
            if self.optimize_restarts == 1:
                self.model.optimize(optimizer=self.optimizer, max_iters=self.max_iters, messages=False,
                                    ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                             max_iters=self.max_iters, ipython_notebook=False, verbose=self.verbose,
                                             robust=True, messages=False)

    def _create_model(self,
                      x: np.ndarray,
                      y: np.ndarray):
        """Create model given input data X and output data Y.

        :param x: 2d array of indices of distance builder
        :param y: model fitness scores
        :return:
        """
        # Make sure input data consists only of positive integers.
        assert np.issubdtype(x.dtype, np.integer) and x.min() >= 0

        # Define kernel
        self.input_dim = x.shape[1]
        # TODO: figure out default kernel kernel initialization
        if self.covariance is None:
            assert self.covariance is not None
            # kern = GPy.kern.RBF(self.input_dim, variance=1.)
        else:
            kern = self.covariance.raw_kernel
            self.covariance = None

        # Define model
        noise_var = y.var() * 0.01 if self.noise_var is None else self.noise_var
        normalize = x.size > 1  # only normalize if more than 1 observation.
        self.model = GPRegression(x, y, kern, noise_var=noise_var, normalizer=normalize)

        # Set hyperpriors
        if self.kernel_hyperpriors is not None:
            if 'GP' in self.kernel_hyperpriors:
                # Set likelihood hyperpriors.
                likelihood_hyperprior = self.kernel_hyperpriors['GP']
                set_priors(self.model.likelihood, likelihood_hyperprior, in_place=True)
            if 'SE' in self.kernel_hyperpriors:
                # Set kernel hyperpriors.
                se_hyperprior = self.kernel_hyperpriors['SE']
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

    def update(self, x_all, y_all, x_new, y_new):
        """Update model with new observations."""
        if self.model is None:
            self._create_model(x_all, y_all)
        else:
            self.model.set_XY(x_all, y_all)

        self.train()

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

    def get_f_max(self):
        """
        Returns the location where the posterior mean is takes its maximal value.
        """
        return self.model.predict(self.model.X)[0].max()

    def plot(self, **plot_kwargs):
        import matplotlib.pyplot as plt
        self.model.plot(plot_limits=(0, self.model.kern.n_models - 1), resolution=self.model.kern.n_models,
                        **plot_kwargs)
        plt.show()

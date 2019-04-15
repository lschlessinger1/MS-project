import numpy as np
from GPy import Param
from GPy.kern import Kern
from GPy.kern.src.psi_comp import PSICOMP_RBF_GPU, PSICOMP_RBF, PSICOMP_Linear
from GPy.kern.src.stationary import Stationary
from GPy.util.linalg import tdot
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from scipy.spatial.distance import cdist


class KernelKernel(Stationary):
    """Kernel kernel"""
    _support_GPU = True

    def __init__(self, distance_metric, dm_kwargs_dict=None, input_dim=1, variance=1., lengthscale=None, ARD=False,
                 active_dims=None, name='kernel_kernel', useGPU=False, inv_l=False):
        super(KernelKernel, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param('inv_lengthscale', 1. / self.lengthscale ** 2, Logexp())
            self.link_parameter(self.inv_l)

        self.dist_metric = distance_metric
        if dm_kwargs_dict is None:
            self.dm_kwargs_dict = dict()
        else:
            self.dm_kwargs_dict = dm_kwargs_dict

    ### overrride this

    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X2 is None:
            X2 = X
        r = cdist(X, X2, self.dist_metric, **self.dm_kwargs_dict)
        return r

    ### overrride this
    @Cache_this(limit=3, ignore_args=())
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            # FIXME: currently ARD has no effect because input dim always = 1
            #             if X2 is not None:
            #                 X2 = X2 / self.lengthscale
            #             return self._unscaled_dist(X/self.lengthscale, X2)
            return self._unscaled_dist(X, X2) / self.lengthscale
        else:
            return self._unscaled_dist(X, X2) / self.lengthscale

    ###

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(KernelKernel, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.KernelKernel"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r ** 2 - 1) * self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance  # as the diagonal of r is always filled with zeros

    def __getstate__(self):
        dc = super(KernelKernel, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(KernelKernel, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1  # TODO: higher dim spectra?
        return self.variance * np.sqrt(2 * np.pi) * self.lengthscale * np.exp(-self.lengthscale * 2 * omega ** 2 / 2)

    def parameters_changed(self):
        if self.use_invLengthscale: self.lengthscale[:] = 1. / np.sqrt(self.inv_l + 1e-200)
        super(KernelKernel, self).parameters_changed()

    # ---------------------------------------#
    #             PSI statistics            #
    # ---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z,
                                                                        variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale * (self.lengthscale ** 3 / -2.)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(KernelKernel, self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale: self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(KernelKernel, self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale: self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)


class LinScaleShift(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^{\\text{input_dim}} \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, variances=None, shifts=None, ARD=False, active_dims=None, name='lin'):
        super(LinScaleShift, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
            if shifts is not None:
                shifts = np.asarray(shifts)
                assert shifts.size == 1, "Only one shift needed for non-ARD kernel"
            else:
                shifts = np.ones(1)
        else:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances = np.ones(self.input_dim)
            if shifts is not None:
                shifts = np.asarray(shifts)
                assert shifts.size == self.input_dim, "bad number of shifts, need one ARD shift per input_dim"
            else:
                shifts = np.ones(self.input_dim)

        self.variances = Param('variances', variances, Logexp())
        self.shifts = Param('shifts', shifts, Logexp())
        self.link_parameter(self.variances)
        self.link_parameter(self.shifts)
        self.psicomp = PSICOMP_Linear()

    def to_dict(self):
        input_dict = super(LinScaleShift, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.LinScaleShift"
        input_dict["variances"] = self.variances.values.tolist()
        input_dict["shifts"] = self.shifts.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return LinScaleShift(**input_dict)

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        X_s = X - self.shifts
        if self.ARD:
            if X2 is None:
                return tdot(X_s * np.sqrt(self.variances))
            else:
                X2_s = X2 - self.shifts
                rv = np.sqrt(self.variances)
                return np.dot(X_s * rv, (X2_s * rv).T)
        else:
            if X2 is None:
                X2_s = X_s
            else:
                X2_s = X2 - self.shifts
            return self._dot_product(X_s, X2_s) * self.variances

    @Cache_this(limit=3, ignore_args=(0,))
    def _dot_product(self, X, X2=None):
        if X2 is None:
            return tdot(X)
        else:
            return np.dot(X, X2.T)

    def Kdiag(self, X):
        return np.sum(self.variances * np.square(X - self.shifts), -1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        X_s = X - self.shifts
        if X2 is None:
            dL_dK = (dL_dK + dL_dK.T) / 2
            X2 = X
            X2_s = X_s
        else:
            X2_s = X2 - self.shifts

        if self.ARD:
            if X2 is None:
                # self.variances.gradient = np.array([np.sum(dL_dK * tdot(X[:, i:i + 1])) for i in range(self.input_dim)])
                self.variances.gradient = (dL_dK.dot(X_s) * X_s).sum(0)  # np.einsum('ij,iq,jq->q', dL_dK, X, X)
                self.shifts.gradient = (2 * self.shifts - X - X2).sum(0)
            else:
                # product = X[:, None, :] * X2[None, :, :]
                # self.variances.gradient = (dL_dK[:, :, None] * product).sum(0).sum(0)
                self.variances.gradient = (dL_dK.dot(X2_s) * X_s).sum(0)  # np.einsum('ij,iq,jq->q', dL_dK, X, X2)
        else:
            self.variances.gradient = np.sum(self._dot_product(X_s, X2_s) * dL_dK)
            self.shifts.gradient = np.sum((2 * self.shifts - X - X2) * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        X_s = X - self.shifts
        tmp = dL_dKdiag[:, None] * X_s ** 2
        if self.ARD:
            self.variances.gradient = tmp.sum(0)
        else:
            self.variances.gradient = np.atleast_1d(tmp.sum())

    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            dL_dK = (dL_dK + dL_dK.T) / 2
        if X2 is None:
            X_s = X - self.shifts
            return dL_dK.dot(X_s) * (2 * self.variances)  # np.einsum('jq,q,ij->iq', X, 2*self.variances, dL_dK)
        else:
            # return (((X2[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)
            X2_s = X2 - self.shifts
            return dL_dK.dot(X2_s) * self.variances  # np.einsum('jq,q,ij->iq', X2, self.variances, dL_dK)

    def gradients_XX(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:

        returns the full covariance matrix [QxQ] of the input dimensionfor each pair or vectors, thus
        the returned array is of shape [NxNxQxQ].

        ..math:
            \frac{\partial^2 K}{\partial X2 ^2} = - \frac{\partial^2 K}{\partial X\partial X2}

        ..returns:
            dL2_dXdX2:  [NxMxQxQ] for X [NxQ] and X2[MxQ] (X2 is X if, X2 is None)
                        Thus, we return the second derivative in X2.
        """
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0], X.shape[1], X.shape[1]))
        # if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        # if X2 is None:
        #    return np.ones(np.repeat(X.shape, 2)) * (self.variances[None,:] + self.variances[:, None])[None, None, :, :]
        # else:
        #    return np.ones((X.shape[0], X2.shape[0], X.shape[1], X.shape[1])) * (self.variances[None,:] + self.variances[:, None])[None, None, :, :]

    def gradients_X_diag(self, dL_dKdiag, X):
        X_s = X - self.shifts
        return 2. * self.variances * dL_dKdiag[:, None] * X_s

    def gradients_XX_diag(self, dL_dKdiag, X):
        return np.zeros((X.shape[0], X.shape[1], X.shape[1]))

        # dims = X.shape
        # if cov:
        #    dims += (X.shape[1],)
        # return 2*np.ones(dims)*self.variances

    def input_sensitivity(self, summarize=True):
        return np.ones(self.input_dim) * self.variances

    # ---------------------------------------#
    #             PSI statistics            #
    # ---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[
            0]
        if self.ARD:
            self.variances.gradient = dL_dvar
        else:
            self.variances.gradient = dL_dvar.sum()

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[1]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2:]

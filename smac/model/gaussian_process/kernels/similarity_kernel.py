from __future__ import annotations

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior


class SimilarityKernel(AbstractKernel, kernels.Kernel):
    """
    Indicator-based similarity kernel for categorical features.
    
    The Implementation is based on the paper "Bayesian Optimisation over Multiple Continuous and Categorical Inputs"
    (Section 5.2): https://arxiv.org/pdf/1906.08878.pdf.
        
    The formula of the SimilarityKernel is:
        - K(h,h') = sigma / c * sum_i=1^c I(h_i - h'_i)
        
    Parameters
    ----------
    noise_level : float
        The sigma in the kernel formula
    noise_level_bounds: tuple[float, float] | list[tuple[float, float]]
        The range of possible values for noise_level
    operate_on : np.ndarray
        On which numpy array should be operated on
    has_conditions : bool
        Whether the kernel has conditions
    prior : AbstractPrior
        Which prior the kernel is using
    """

    def __init__(
            self,
            noise_level: float = 1.0,
            noise_level_bounds: tuple[float, float] | list[tuple[float, float]] = (1e-5, 1e5),
            operate_on: np.ndarray | None = None,
            has_conditions: bool = False,
            prior: AbstractPrior | None = None,
    ):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
        )

    @property
    def hyperparameter_noise_level(self) -> kernels.Hyperparameter:
        return kernels.Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def diag(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[np.newaxis, :]
        return self.noise_level * np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        return True

    def _call(
            self,
            X: np.ndarray,
            Y: np.ndarray | None = None,
            eval_gradient: bool = False,
            active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[np.newaxis, :]

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.mean(X[:, np.newaxis, :] == X[np.newaxis, :, :], axis=-1)
        else:
            K = self.noise_level * np.mean(X[:, np.newaxis, :] == Y[np.newaxis, :, :], axis=-1)

        if active is not None:
            K = K * active

        if eval_gradient:
            if not self.hyperparameter_noise_level.fixed:
                K_grad = np.mean(X[:, np.newaxis, :] == X[np.newaxis, :, :], axis=-1)
                return K, K_grad[:, :, np.newaxis]
            else:
                return K, np.empty(shape=(X.shape[0], X.shape[0], self.n_dims))
        return K

    def __repr__(self) -> str:
        return "{0}(noise_level={1:.3g})".format(
            self.__class__.__name__, np.ravel(self.noise_level)[0]
        )

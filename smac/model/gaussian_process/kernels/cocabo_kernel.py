from __future__ import annotations

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels import RBFKernel
from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.kernels.similarity_kernel import SimilarityKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior


class CoCaBOKernel(AbstractKernel, kernels.KernelOperator):
    """
    CoCaBO kernel for categorical + continuous features.
    
    The Implementation is based on the paper "Bayesian Optimisation over Multiple Continuous and Categorical Inputs"
    (Section 5.2): https://arxiv.org/pdf/1906.08878.pdf.
        
    The CoCaBOKernel is a weight-based kernel which divide the input Z into the categorical H and continuous input X.

    The formula of the CoCaBOKernel is:
        - K(z, z') = (1 - weight) * ((k_h(h, h') + k_x(x, x')) + weight * (k_h(h, h') * k_x(x, x'))

    If no categorical features are available, the kernel simplifies to:
        - K(z, z') = k_x(z, z')
        
    Parameters
    ----------
    k1 : kernels.Kernel
        The kernel k_h(h) for the categorical features
    k2 : kernels.Kernel
        The kernel k_x(x) for the continuous features
    weight : float
        The weight for controlling the sum and product of the two kernels
    weight_bounds : tuple[float, float] | list[tuple[float, float]]
        The range of the weight parameter
    operate_on : np.ndarray
        On which numpy array should be operated on
    has_conditions : bool
        Whether the kernel has conditions
    prior : AbstractPrior
        Which prior the kernel is using
    """

    def __init__(
            self,
            k1: kernels.Kernel = SimilarityKernel(),
            k2: kernels.Kernel = RBFKernel(),
            weight: float = np.exp(0.5),
            weight_bounds: tuple[float, float] | list[tuple[float, float]] = (1.0, np.exp(1.0)),
            operate_on: np.ndarray | None = None,
            has_conditions: bool = False,
            prior: AbstractPrior | None = None,
    ) -> None:
        self.weight = weight
        self.log_weight = np.log(weight)
        self.weight_bounds = weight_bounds
        self.has_conditions = has_conditions
        self.operate_on = operate_on
        self.prior = prior
        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
            k1=k1,
            k2=k2,
        )

    @property
    def hyperparameter_weight(self) -> kernels.Hyperparameter:
        return kernels.Hyperparameter("weight", "numeric", self.weight_bounds)

    @property
    def hyperparameters(self) -> list[kernels.Hyperparameter]:
        r = [kernels.Hyperparameter(
            self.hyperparameter_weight.name,
            self.hyperparameter_weight.value_type,
            self.hyperparameter_weight.bounds,
            self.hyperparameter_weight.n_elements,
        )]
        for hyperparameter in self.k1.hyperparameters:
            r.append(
                kernels.Hyperparameter(
                    "k1__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        for hyperparameter in self.k2.hyperparameters:
            r.append(
                kernels.Hyperparameter(
                    "k2__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self) -> np.ndarray:
        return np.append(np.array(self.log_weight), [self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        k1_dims = self.k1.n_dims
        self.log_weight = theta[0]
        self.weight = np.exp(theta[0])
        self.k1.theta = theta[1:k1_dims + 1]
        self.k2.theta = theta[k1_dims + 1:]

    @property
    def bounds(self) -> np.ndarray:
        return super(kernels.KernelOperator, self).bounds

    def _separate_X(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Separates the feature matrix X into H, X', where (...)
            - H contains all columns of the categorical features
            - X' contains all columns of the continuous/discrete features

        Parameters
        ----------
        X : np.ndarray [N, D]
            The feature vector we want to separate

        Returns
        -------
        np.ndarray [N, D1]
            The feature matrix H
        
        np.ndarray [N, D2]
            The feature matrix X'
        """
        
        # Determine the data types of each column
        categorical_columns = np.zeros(shape=len(X.T), dtype=bool)
        for i, col in enumerate(X.T):
            converted_col = np.array(col).astype(int)
            if np.allclose(col, converted_col):
                # Case: Column is categorical
                categorical_columns[i] = True
        X_categorical = X[:, categorical_columns].astype(int)
        X_continuous = X[:, ~categorical_columns].astype(float)
        return X_categorical, X_continuous
    
    def get_params(self, deep=True) -> dict:
        params = super(AbstractKernel, self).get_params(deep)

        # Append own params
        params["weight"] = self.weight
        params["weight_bounds"] = self.weight_bounds
        params["has_conditions"] = self.has_conditions
        params["operate_on"] = self.operate_on
        params["prior"] = self.prior
        return params

    def diag(self, X: np.ndarray) -> np.ndarray:
        # Separate X between categorical H and numerical X'
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_categorical, X_continuous = self._separate_X(X)
        
        if X_categorical.size != 0 and X_continuous.size != 0:
            # Case: Categorical and continuous features are available
            # Compute the combined diagonal
            K1 = self.k1.diag(X_categorical)
            K2 = self.k2.diag(X_continuous)
            return self.log_weight * K1 * K2 + (1 - self.log_weight) * (K1 + K2)
        elif X_categorical.size != 0:
            # Case: Only categorical features are available
            return self.k1.diag(X_categorical)
        else:
            # Case: Only continuous features are available
            return self.k2.diag(X_continuous)

    def is_stationary(self) -> bool:
        return self.k1.is_stationary() and self.k2.is_stationary()

    def _call(
            self,
            X: np.ndarray,
            Y: np.ndarray | None = None,
            eval_gradient: bool = False,
            active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # Separate X between categorical H and numerical X'
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_categorical, X_continuous = self._separate_X(X)

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            if X_categorical.size != 0 and X_continuous.size != 0:
                # Case: Categorical and continuous features are available 
                K1, K1_grad = self.k1(X_categorical, eval_gradient=True)
                K2, K2_grad = self.k2(X_continuous, eval_gradient=True)
                K = self.log_weight * K1 * K2 + (1 - self.log_weight) * (K1 + K2)
            elif X_categorical.size != 0:
                # Case: Only categorical features are available
                K1, K1_grad = self.k1(X_categorical, eval_gradient=True)
                K = K1
            else:
                # Case: Only continuous features are available
                K2, K2_grad = self.k2(X_continuous, eval_gradient=True)
                K = K2
        else:
            # Separate Y into categorical and continuous
            Y_categorical, Y_continuous = self._separate_X(Y)
            
            if X_categorical.size != 0 and X_continuous.size != 0:
                # Case: Categorical and continuous features are available 
                K1 = self.k1(X_categorical, Y_categorical, eval_gradient=False)
                K2 = self.k2(X_continuous, Y_continuous, eval_gradient=False)
                K = self.log_weight * K1 * K2 + (1 - self.log_weight) * (K1 + K2)
            elif X_categorical.size != 0:
                # Case: Only categorical features are available
                K = self.k1(X_categorical, Y_categorical, eval_gradient=False)
            else:
                # Case: Only continuous features are available
                K = self.k2(X_continuous, Y_continuous, eval_gradient=False)
            
        if active is not None:
            K = K * active

        if eval_gradient:
            if not self.hyperparameter_weight.fixed:

                if X_categorical.size != 0 and X_continuous.size != 0:
                    # Case: Categorical and continuous features are available 
                    K1_ext = np.repeat(K1[:, :, np.newaxis], self.k2.n_dims, axis=-1)
                    K2_ext = np.repeat(K2[:, :, np.newaxis], self.k1.n_dims, axis=-1)

                    # Compute the gradients dkz/dweight (formula 17 in the paper)
                    K_grad_weight = (-(K1 + K2) + K1 * K2)[:, :, np.newaxis]
                    
                    # Compute the gradients dkz/dtheta_h (formula 15 in the paper)
                    K_grad_h = (1 - self.log_weight) * K1_grad + self.log_weight * K2_ext * K1_grad
                    
                    # Compute the gradients dkz/dtheta_h (formula 16 in the paper)
                    K_grad_x = (1 - self.log_weight) * K2_grad + self.log_weight * K1_ext * K2_grad
                elif X_categorical.size != 0:
                    # Case: Only categorical features are available
                    # Compute the gradients dkz/weight (becomes zero in this case)
                    K_grad_weight = np.zeros(shape=(X_categorical.shape[0], X_categorical.shape[0], 1))
                    
                    # Compute the gradients dkz/dtheta_h (becomes equal to dkh/dtheta_h)
                    K_grad_h = K1_grad
                    
                    # Compute the gradients of dkz/dtheta_x (becomes zero in this case)
                    K_grad_x = np.zeros(shape=(X_categorical.shape[0], X_categorical.shape[0], self.k2.n_dims))
                else:
                    # Case: Only continuous features are available
                    # Compute the gradients dkz/weight (becomes zero in this case)
                    K_grad_weight = np.zeros(shape=(X_continuous.shape[0], X_continuous.shape[0], 1))
                    
                    # Compute the gradients dkz/dtheta_h (becomes zero in this case)
                    K_grad_h = np.zeros(shape=(X_continuous.shape[0], X_continuous.shape[0], self.k1.n_dims))
                    
                    # Compute the gradients of dkz/dtheta_x (becomes equal to dkx/dtheta_x)
                    K_grad_x = K2_grad
                
                # Concatenate the gradients together
                K_grad = np.concatenate((K_grad_weight, K_grad_h, K_grad_x), axis=-1)
                return K, K_grad
            else:
                return K, np.zeros(shape=(X.shape[0], X.shape[0], self.n_dims))
        return K

from __future__ import annotations

import warnings
from typing import Any, Iterator

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.preprocessing import MinMaxScaler

from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
from smac.model.abstract_model import AbstractModel
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class MultiMAB(AbstractAcquisitionFunction):
    """Implementation of Multi-Agent Multi-Armed Bandit Acquisition function. The MultiMAB optimizes all categorical
    hyperparameters of a configuration space by using for each categorical hyperparameter a separate MAB and optimizing
    its weights with the EXP3 algorithm.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space where we contain categorical, continuous hyperparameters
    gamma : float
        The factor to control exploration-exploitation in EXP3 for each MAB.
        gamma == 1: Only Exploration, No Exploitation
        gamma == 0: No Exploration, Only Exploitation 
    seed : int
        The random seed for each MAB
    """

    def __init__(
            self,
            configspace: ConfigurationSpace,
            gamma: float,
            seed: int,
    ) -> None:
        super(MultiMAB, self).__init__()
        self._configspace = configspace

        # Initialize the MABs
        self._multi_mab = [
            MAB(index=i, K=len(hp.choices), gamma=gamma, seed=seed) for i, hp in enumerate(self._configspace.values())
            if isinstance(hp, CategoricalHyperparameter)
        ]

    @property
    def name(self) -> str:
        return "Multi-Multi-Armed Bandit"

    def update(self, model: AbstractModel, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        for mab in self._multi_mab:
            mab.update(model=model, X=X, Y=Y, **kwargs)

    def get_hp_values(self, actions: np.ndarray) -> np.ndarray:
        """Returns the corresponding hyperparameter values, given the actions as indices for each hyperparameter value.

        Parameters
        ----------
        actions : np.ndarray [1, NUM_MABS]
            The actions for each MAB as indices.

        Returns
        -------
        np.ndarray [1, NUM_MABS]
            The corresponding values of each categorical hyperparameter
        """
        if actions.shape != (1, len(self._multi_mab)):
            raise ValueError("MultiMAB only support get_hp_values(...) with shape (1, NUM_MABS)!")

        # Get the categorical values for each categorical hyperparameter
        hp_values = np.zeros(shape=(1, len(self._multi_mab)), dtype=object)
        for i, mab in enumerate(self._multi_mab):
            hp = list(self._configspace.values())[mab._index]
            hp_values[0, i] = hp.choices[actions[0, i]]
        return hp_values

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        if len(configurations) != 1:
            raise ValueError("MultiMAB only support __call__(...) with shape (1, N)!")

        # Get the indices of each MAB
        indices = [mab._index for mab in self._multi_mab]

        # Select only the categorical values of each MAB
        X = convert_configurations_to_array(configurations)[:, indices]

        # Reshape from (NUM_MABS,) to (1, NUM_MABS)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # Compute the acquisition values (as indices)
        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max
        return acq

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if X.shape != (1, len(self._multi_mab)):
            raise ValueError("MultiMAB only support _compute(...) with shape (1, NUM_MABS)!")

        # Sample for the given input
        indices = np.zeros(shape=X.shape, dtype=int)

        # Sample for each mab
        for i, mab in enumerate(self._multi_mab):
            # Select the corresponding categorical value for the given MAB
            x = X[:, i]

            # Reshape from (1,) to (1, 1)
            if x.ndim == 1:
                x = x[np.newaxis, :]

            # Sample the next action from the MAB
            indices[:, i] = mab._compute(x)

        return indices

    def __getitem__(self, index: int) -> MAB:
        return self._multi_mab[index]

    def __setitem__(self, index: int, value: MAB) -> None:
        self._multi_mab[index] = value

    def __iter__(self) -> Iterator:
        return iter(self._multi_mab)

    def __str__(self) -> str:
        header = "MultiMAB(mabs=[\n"
        lines = [f"\t{mab},\n" for mab in self._multi_mab]
        end = "])"
        return "".join([header, *lines, end])

    def __repr__(self) -> str:
        return self.__str__()


class MAB(AbstractAcquisitionFunction):
    """Implementation of Multi-Armed Bandit Acquisition function. The Multi-Armed Bandit optimizes one categorical hyperparameter of
        a configuration space by optimizing its sampling weights with the EXP3 algorithm.
        
        The implementation is based on the EXP3 algorithm:
        https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

        Parameters
        ----------
        index : int
            The index of the categorical hyperparameter
        K : int
            The number of values (choices) for the categorical hyperparameter
        gamma : float
            The factor to control exploration-exploitation in EXP3 for each MAB.
            gamma == 1: Only Exploration, No Exploitation
            gamma == 0: No Exploration, Only Exploitation 
        seed : int
            The seed for the random number generator
        """

    def __init__(self, index: int, K: int, gamma: float, seed: int):
        super(MAB, self).__init__()
        self._index = index
        self._K = K
        self._gamma = gamma
        self._rng = np.random.RandomState(seed)

        # Initialize the uniform wights
        self._weights = np.array([1.0 for _ in range(K)])

        # Initialize the last action
        self._last_action = -1
        self._first_time = True

    @property
    def name(self) -> str:
        return "Multi-Armed Bandit"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "K": self._K,
                "gamma": self._gamma,
            }
        )
        return meta

    def _normalize(self, weights: np.ndarray, axis: int = 0) -> np.ndarray:
        """Normalizes the given weight vector/matrix into a probability distribution.

            Parameters
            ----------
            weights : np.ndarray [N,]
                The weight vector we want to normalize
            
            axis : int
                The axis where we want to compute the probability distribution

            Returns
            -------
            np.ndarray [N,]
                The (normalized) probability distribution along the given axis
            """
        return weights / np.sum(weights, axis=axis)

    @property
    def _prob(self) -> np.ndarray:
        """The probability distribution p(t)=[p_1(t), ..., p_K(t)] according to EXP3 algorithm.

            Returns
            -------
            np.ndarray [N,] or [N, D]
                The probability distribution p(t) with the EXP3 algorithm
            """
        # Normalize the weights to a probability distribution
        probs = self._normalize(self._weights)

        # Shift p(t) according to the EXP3 algorithm
        probs = (1 - self._gamma) * probs + (self._gamma / self._K)

        # Apply normalization of the probabilities
        probs = self._normalize(probs)
        return probs

    def _update(self, **kwargs: Any) -> None:
        if self._last_action == -1 and not self._first_time:
            # Case: last action was not set correctly
            warnings.warn(
                "Use __call__(...) to set the last action before updating the weights of the MAB!",
                UserWarning
            )
            return

        # Set first_time to False
        self._first_time = False
        
        # Perform a min-max normalization over the y-values
        # Get the last observed y-values for the last action
        reward = MinMaxScaler().fit_transform(self.Y)[-1].item()

        # Compute the estimated reward
        estimated_reward = reward / max(1e-10, self._prob[self._last_action])

        # Update the weight of the last action taken after the EXP3 algorithm
        self._weights[self._last_action] = self._weights[self._last_action] * np.exp(
            -self._gamma * estimated_reward / self._K)

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        if len(configurations) != 1:
            raise ValueError("MAB only support __call__(...) with shape (1, N)!")

        # Get the value of the categorical hyperparameter
        X = convert_configurations_to_array(configurations)[:, self._index]

        # Adjust from shape (1,) to (1, 1)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        
        # Compute the acquisition values (indices of next actions)
        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max

        return acq

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if X.shape != (1, 1):
            raise ValueError("MAB only support _compute(...) with shape (1, 1)!")

        # Sample a choice of the categorical value as index
        test = self._normalize(self._weights)
        index = self._rng.choice(self._K, size=X.shape[0], p=test)#  p=self._prob)

        # Update last action to the current index
        self._last_action = index.item()

        return index

    def __str__(self) -> str:
        return f"MAB(index={self._index}, K={self._K}, gamma={self._gamma}, weights={self._weights})"

    def __repr__(self) -> str:
        return self.__str__()

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from sklearn.preprocessing import MinMaxScaler

from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
from smac.model.abstract_model import AbstractModel
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class MultiMAB(AbstractAcquisitionFunction):
    """
    Implementation of Multi-Agent Multi-Armed Bandit Acquisition function. The MultiMAB optimizes all categorical
    hyperparameters of a configuration space by using for each categorical hyperparameter a separate MAB and optimizing
    its weights with the EXP3 algorithm.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space where we contain categorical, continuous hyperparameters
    gamma : float
        The factor to control exploration-exploitation in EXP3 for each MAB
        If gamma == 1: Only Exploration, No Exploitation
        If gamma == 0: No Exploration, Only Exploitation
    seed : int
        The random seed for each MAB
    """

    def __init__(
            self,
            configspace: ConfigurationSpace,
            gamma: float,
            seed: int,
    ):
        super(MultiMAB, self).__init__()
        self._configspace = configspace
        self._rng = np.random.RandomState(seed)
        # Initialize the MABs
        self._multi_mab = []
        for i, hp in enumerate(self._configspace.values()):
            if isinstance(hp, CategoricalHyperparameter):
                new_seed = self._rng.randint(0, int(1e7))
                self._multi_mab.append(MAB(index=i, K=len(hp.choices), gamma=gamma, seed=new_seed))
            elif isinstance(hp, OrdinalHyperparameter):
                new_seed = self._rng.randint(0, int(1e7))
                self._multi_mab.append(MAB(index=i, K=len(hp.sequence), gamma=gamma, seed=new_seed))

    @property
    def name(self) -> str:
        return "Multi-MAB"

    def update(self, model: AbstractModel, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        for mab in self._multi_mab:
            mab.update(model=model, X=X, Y=Y, **kwargs)

    def get_hp_values(self, actions: np.ndarray) -> np.ndarray:
        """
        Returns the corresponding hyperparameter values, given the actions as indices for each hyperparameter value.

        Parameters
        ----------
        actions : np.ndarray [1, NUM_MABS]
            The actions for each MAB as indices.

        Returns
        -------
        np.ndarray [1, NUM_MABS]
            The corresponding values of each categorical hyperparameter
        """
        if actions.shape[1] != len(self._multi_mab):
            raise ValueError("MultiMAB only support get_hp_values(...) with shape (NUM_CFGS, NUM_MABS)!")

        # Get the categorical values for each categorical hyperparameter
        hp_values = np.zeros(shape=actions.shape, dtype=object)
        for i, action in enumerate(actions):
            for j, mab in enumerate(self._multi_mab):
                hp = list(self._configspace.values())[mab.index]
                if isinstance(hp, OrdinalHyperparameter):
                    hp_values[i, j] = hp.sequence[action[j]]
                elif isinstance(hp, CategoricalHyperparameter):
                    hp_values[i, j] = hp.choices[action[j]]
        return hp_values

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        # Get the indices of each MAB
        indices = [mab.index for mab in self._multi_mab]

        # Select only the categorical values of each MAB
        X = convert_configurations_to_array(configurations)[:, indices]

        # Reshape from (1,) to (1, 1)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # Compute the acquisition values (as indices)
        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max
        return acq

    def _compute(self, X: np.ndarray) -> np.ndarray:
        # Sample for the given input
        next_actions = np.zeros(shape=X.shape, dtype=int)

        # Sample for each mab
        for i, mab in enumerate(self._multi_mab):
            # Sample the next action from the MAB
            next_actions[:, i] = mab._compute(X[:, i, np.newaxis]).reshape(-1)

        return next_actions

    def __len__(self) -> int:
        return len(self._multi_mab)

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
    """
    Implementation of Multi-Armed Bandit Acquisition function.
    
    The Multi-Armed Bandit optimizes one categorical hyperparameter of a configuration space by optimizing its sampling
    weights with the EXP3 algorithm.

    The implementation is based on the EXP3 algorithm:
    https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

    Because the EXP3 algorithm is designed for maximization problems and to handle rewards in range [0, 1], our
    algorithm does two extensions to apply it for minimization problems.
    1. It does a min-max normalization over the entire dataset (X, y) of the model to range the values inside [0, 1]
    2. Multiply -1 to the rewards to convert it to a maximization problem

    Parameters
    ----------
    index : int
        The index of the categorical hyperparameter (from the configuration space)
    K : int
        The number of values (choices) for the categorical hyperparameter
    gamma : float
        The factor to control exploration-exploitation in EXP3 for each MAB
        If gamma == 1: Only Exploration, No Exploitation
        If gamma == 0: No Exploration, Only Exploitation
    seed : int
        The seed for the random number generator
    """

    def __init__(self, index: int, K: int, gamma: float, seed: int):
        super(MAB, self).__init__()
        self._index = index
        self._K = K
        self._gamma = gamma
        self._rng = np.random.RandomState(seed)

        # EXP3:
        # 1. Initialize the weights w_i(1) = 1 for i = 1, ..., K
        self._weights = np.array([1.0 for _ in range(K)])

        # Initialize the last action
        self._history = 0

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

    @property
    def index(self) -> int:
        """Returns the index of the categorical hyperparameter."""
        return self._index

    def _normalize_weights(self) -> np.ndarray:
        """
        Normalizes the weight vector into a probability distribution.

        Returns
        -------
        np.ndarray [N,]
            The (normalized) probability distribution along the given axis
        """
        return self._weights / np.sum(self._weights)

    @property
    def _prob(self) -> np.ndarray:
        """
        Computes the probability distribution p_i(t) according to the EXP3 algorithm.
        
        The probability distribution is computed by the following formula:
        p_i(t) = (1 - gamma) * w_i(t) / sum_j=1^K w_j(t) + gamma / K

        Returns
        -------
        np.ndarray [N,]
            The probability distribution p(t)=[p_1(t), p_2(t), ...]
        """
        # Normalize the weights to a probability distribution w_i(t) / sum_j=1^K w_j(t)
        weights = self._normalize_weights()

        # Compute the probability distribution  p_i(t) = (1 - gamma) * w_i(t) / sum_j=1^K w_j(t) + gamma / K
        probs = (1 - self._gamma) * weights + (self._gamma / self._K)

        return probs

    def _update(self, **kwargs: Any) -> None:
        # EXP3:
        # 2.3. Observe reward x_i_t(t)
        # Because the range of the value can be [-inf, +inf] we perform a min-max normalization to scale it to [0, 1]
        rewards = MinMaxScaler().fit_transform(self.Y).reshape(-1)

        for t in range(self._history, self.X.shape[0]):
            # EXP3:
            # 2.3. Observe reward x_i_t(t)
            # Get the action and reward from configuration t
            action = int(self.X[t, self._index])
            reward = float(rewards[t])

            # EXP3:
            # 2.4. Define the estimated reward x^_i_t(t) = x_i_t(t) / p_i_t(t)
            estimated_reward = reward / max(1e-10, self._prob[action])

            # EXP3:
            # 2.5.: Update weights w_i_t(t+1) = w_i_t(t) * exp(gamma * x^_i_t(t) / K)
            # Because we have a minimization problem we use -x^_i_t(t) for the update
            self._weights[action] = (
                    self._weights[action] * np.exp(-self._gamma * estimated_reward / self._K)
            )
    
        # Update history
        self._history = len(self.X)

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        # Get the value of the categorical hyperparameter
        X = convert_configurations_to_array(configurations)[:, self._index]

        # Adjust from shape (N,) to (N, 1)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        # Compute the acquisition values (indices of next actions)
        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max

        return acq

    def _compute(self, X: np.ndarray) -> np.ndarray:
        # EXP3:
        # 2.2 Draw the next action i_t (as index) according to the distribution of p_i(t)
        next_action = self._rng.choice(self._K, size=X.shape, p=self._prob, replace=True)
        return next_action

    def __str__(self) -> str:
        return f"MAB(index={self._index}, K={self._K}, gamma={self._gamma}, weights={self._weights})"

    def __repr__(self) -> str:
        return self.__str__()

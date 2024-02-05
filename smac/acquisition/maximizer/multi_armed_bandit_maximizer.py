from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function import MultiMAB
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionFunction,
    AbstractAcquisitionMaximizer
)
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class MultiMABMaximizer(AbstractAcquisitionMaximizer):
    """
    The MultiMABMaximizer uses a combination of Random Search (RS) and Multi-Agent MABs to sample new configurations.
    First the Multi-Agent MAB selects for each categorical hyperparameter their next values.
    Secondly we use RS to sample different values for each continuous hyperparameter.
    Additionally, we sort them in desc order to the values of the given acquisition function.
    
    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space where we contain categorical, continuous hyperparameters
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function for the predicting the performance of the continuous hyperparameters
    gamma : float
        The factor to control exploration-exploitation in EXP3 for the Multi-Agent MAB
        If gamma == 1: Only Exploration, No Exploitation
        If gamma == 0: No Exploration, Only Exploitation
    challengers : int
        The number of configurations to be evaluated by the acquisition function
    seed : int
        The random seed for the Multi-Agent MAB
    """
    
    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        gamma: float = 0.5,
        challengers: int = 5000,
        seed: int = 0,
    ):
        super(MultiMABMaximizer, self).__init__(
            configspace=configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        # Initialize the Multi-Agent MAB
        self._mabs = MultiMAB(
            configspace=configspace,
            gamma=gamma,
            seed=seed,
        )
        
        self._random_search = RandomSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            seed=seed,
        )
    
    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        self._random_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "random_search": self._random_search.meta,
            }
        )

        return meta
    
    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False,
    ) -> list[tuple[float, Configuration]]:    
        # Get configurations from random search
        next_configs_by_random_search = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=n_points,
            _sorted=_sorted,
        )
        next_configs_by_random_search = [cfg for _, cfg in next_configs_by_random_search]
        
        if self._mabs:
            # Case: No categorical hyperparameter available in the configuration space
            # Update the Multi-Agent MAB
            self._mabs.update(
                model=self.acquisition_function.model,
                X=self.acquisition_function.X,
                Y=self.acquisition_function.Y
            )

            # Select the next values for each categorical hyperparameter
            next_actions = self._mabs([self._configspace.sample_configuration()])
            hp_values = self._mabs.get_hp_values(next_actions)
            
            # Get the categorical hyperparameter names from the configuration space
            hp_names = list(self._configspace.keys())
            hp_names = [hp_names[mab.index] for mab in self._mabs]

            # Replace the categorical hyperparameter with the MAB ones
            for cfg in next_configs_by_random_search:
                for i, name in enumerate(hp_names):
                    cfg[name] = hp_values[0][i]

        if _sorted:
            # Case: Sort them by the acquisition function values
            for i in range(len(next_configs_by_random_search)):
                next_configs_by_random_search[i].origin = \
                    "Acquisition Function Maximizer: Multi-MAB + Random Search (sorted)"

            return self._sort_by_acquisition_value(next_configs_by_random_search)
        else:
            # Case: Do not sort them by the acquisition function values
            for i in range(len(next_configs_by_random_search)):
                next_configs_by_random_search[i].origin = \
                    "Acquisition Function Maximizer: Multi-MAB + Random Search"

            return [(0, next_configs_by_random_search[i]) for i in range(len(next_configs_by_random_search))]

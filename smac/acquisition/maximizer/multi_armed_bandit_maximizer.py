from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function import MultiMAB
from smac.acquisition.maximizer.abstract_acqusition_maximizer import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acqusition_maximizer import AbstractAcquisitionMaximizer
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

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
        The factor to control exploration-exploitation in EXP3 for the Multi-Agent MAB.
        gamma == 1: Only Exploration, No Exploitation
        gamma == 0: No Exploration, Only Exploitation
    challengers : int
        The number of configurations to be evaluated by the acquisition function.
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

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False,
    ) -> list[tuple[float, Configuration]]:
        # Update the Multi-Agent MAB
        self._mabs.update(
            model=self.acquisition_function.model,
            X=self.acquisition_function.X,
            Y=self.acquisition_function.Y
        )
        
        # Select the next values for each categorical hyperparameter
        actions = self._mabs([self._configspace.sample_configuration()])
        hp_values = self._mabs.get_hp_values(actions)
        
        # Get the categorical hyperparameter names from a configuration space
        hp_names = list(self._configspace.keys())
        hp_names = [hp_names[mab._index] for mab in self._mabs._multi_mab]
        
        # Sample randomly the configuration
        if n_points > 1:
            rand_configs = self._configspace.sample_configuration(size=n_points)
        else:
            rand_configs = [self._configspace.sample_configuration(size=1)]
        
        # Replace the categorical hyperparameter with the MAB ones
        for cfg in rand_configs:
            for i, name in enumerate(hp_names):
                cfg[name] = hp_values[0][i]

        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Acquisition Function Maximizer: Multi-MAB + Random Search (sorted)"
            return self._sort_by_acquisition_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Acquisition Function Maximizer: Multi-MAB + Random Search"

            return [(0, rand_configs[i]) for i in range(len(rand_configs))]
import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.acquisition.function import EI
from smac.acquisition.maximizer import MultiMABMaximizer
from smac.model.gaussian_process import CoCaBOGaussianProcess
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array


# --------------------------------------------------------------
# Test MultiMABMaximizer
# --------------------------------------------------------------
class TestMultiMABMaximizer(unittest.TestCase):

    def setUp(self):
        # Define ConfigurationSpace, X, y
        self.cs = ConfigurationSpace(
            space={
                "A": (0.1, 1.5),
                "B": (2, 10),
                "C": ["c1", "c2", "c3", "c4"],
                "D": ["d1", "d2", "d3"],
            },
            seed=0,
        )
        self.configurations = self.cs.sample_configuration(5)
        self.X = convert_configurations_to_array(self.configurations)
        self.y = np.array([1, 1, 2, 2, 4])

        # Initialize and train the model
        self.kernel = CoCaBOKernel(SimilarityKernel(), RBFKernel(), weight=0.5)
        self.model = CoCaBOGaussianProcess(
            configspace=self.cs,
            kernel=self.kernel,
            n_restarts=10,
            seed=0,
        )
        self.model.train(self.X, self.y)

        # Initialize the acquisition function
        self.ei = EI()
        self.ei._model = self.model
        self.ei._eta = 0

        # Initialize the acquisition function maximizer
        self.mab_maximizer = MultiMABMaximizer(
            configspace=self.cs,
            acquisition_function=self.ei,
            gamma=0.5,
            seed=0,
        )

    def test_maximize_non_sorted(self):
        """
        Tests the method _maximize() with _sorted=False.
        """
        # Get the next configurations based on the maximizer
        configurations = self.mab_maximizer._maximize(
            previous_configs=None,
            n_points=10,
            _sorted=False,
        )

        # Check if all configurations has the same value for categorical hyperparameters
        self.assertTrue(all(cfg["C"] == "c2" for _, cfg in configurations))
        self.assertTrue(all(cfg["D"] == "d2" for _, cfg in configurations))

    def test_maximize_sorted(self):
        """
        Tests the method _maximize() with _sorted=True.
        """
        # Get the next configurations based on the maximizer
        configurations = self.mab_maximizer._maximize(
            previous_configs=None,
            n_points=10,
            _sorted=True,
        )

        # Check if all configurations has the same value for categorical hyperparameters
        self.assertTrue(all(cfg["C"] == "c2" for _, cfg in configurations))
        self.assertTrue(all(cfg["D"] == "d2" for _, cfg in configurations))

        # Check if the acquisition function values are in descending order
        self.assertTrue(all(configurations[i][0] >= configurations[i + 1][0]) for i in range(len(configurations) - 1))


if __name__ == "__main__":
    unittest.main()

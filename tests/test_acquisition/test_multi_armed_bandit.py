import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.acquisition.function import MultiMAB, MAB
from smac.model.gaussian_process import CoCaBOGaussianProcess
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array


# --------------------------------------------------------------
# Test MultiMAB
# --------------------------------------------------------------
class TestMultiMAB(unittest.TestCase):

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
        self.y = np.array([1, 1, 2, 2, 4]).reshape(-1, 1)

        # Initialize the MultiMAB
        self.mabs = MultiMAB(
            configspace=self.cs,
            gamma=0.5,
            seed=0,
        )

        # Initialize the model
        self.kernel = CoCaBOKernel(SimilarityKernel(), RBFKernel(), weight=0.5)
        self.model = CoCaBOGaussianProcess(
            configspace=self.cs,
            kernel=self.kernel,
            n_restarts=10,
            seed=0,
        )

    def test_update(self):
        """
        Tests the method update().
        """
        # Train the model beforehand
        self.model.train(self.X, self.y)

        # Update the mabs 10x times with different last actions
        for _ in range(10):
            # Use the mab for the first time
            self.mabs([self.configurations[0]])

            # Update the mab
            self.mabs.update(model=self.model, X=self.X, Y=self.y)

        np.testing.assert_almost_equal(np.array([0.4920554, 0.3815996, 0.1633308, 0.2753873]), self.mabs[0]._weights)
        np.testing.assert_almost_equal(np.array([0.3262476, 0.181825, 0.1471832]), self.mabs[1]._weights)

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        # Compute the index of the next configuration
        indices = self.mabs([self.configurations[0]])
        np.testing.assert_almost_equal(np.array([[2, 1]]), indices)

    def test_get_hp_values(self):
        """
        Tests the method get_hp_values().
        """
        hp_values = self.mabs.get_hp_values(np.array([[2, 1]]))
        np.testing.assert_equal(np.array([["c3", "d2"]]), hp_values)


# --------------------------------------------------------------
# Test MAB
# --------------------------------------------------------------

class TestMAB(unittest.TestCase):

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
        self.y = np.array([1, 1, 2, 2, 4]).reshape(-1, 1)

        # Initialize the MAB
        self.mab = MAB(
            index=2,
            K=len(self.cs["C"].choices),
            gamma=0.5,
            seed=0,
        )

        # Initialize the model
        self.kernel = CoCaBOKernel(SimilarityKernel(), RBFKernel(), weight=0.5)
        self.model = CoCaBOGaussianProcess(
            configspace=self.cs,
            kernel=self.kernel,
            n_restarts=10,
            seed=0,
        )

    def test_normalize(self):
        """
        Tests the method normalize().
        """
        weights = [1.0, 1.0, 1.0]
        normalized_weights = self.mab._normalize(weights)
        np.testing.assert_almost_equal(np.array([0.3333333, 0.3333333, 0.3333333]), normalized_weights)

    def test_prob(self):
        """
        Tests the property prob.
        """
        np.testing.assert_almost_equal(np.array([0.25, 0.25, 0.25, 0.25]), self.mab._prob)

    def test_update(self):
        """
        Tests the method update().
        """
        # Train the model first
        self.model.train(self.X, self.y)

        # Update the mab 10x times with different last actions
        for _ in range(10):
            # Use the mab for the first time
            self.mab([self.configurations[0]])

            # Update the mab
            self.mab.update(model=self.model, X=self.X, Y=self.y)
        np.testing.assert_almost_equal(np.array([0.4920554, 0.3815996, 0.1633308, 0.2753873]), self.mab._weights)

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        # Compute the index of the next configuration
        index = self.mab([self.configurations[0]])
        np.testing.assert_almost_equal(np.array([2]), index)


if __name__ == "__main__":
    unittest.main()

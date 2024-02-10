import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.acquisition.function import MultiMAB, MAB
from smac.model.gaussian_process import CoCaBOGaussianProcess
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array


class TestMultiMAB(unittest.TestCase):
    """
    Tests the class MultiMAB.
    """

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

        # Update the mab
        self.mabs.update(model=self.model, X=self.X, Y=self.y)

        np.testing.assert_almost_equal(np.array([0.61864039, 1., 0.84648172, 0.8492465]), self.mabs[0]._weights)
        np.testing.assert_almost_equal(np.array([1., 1., 0.40542698]), self.mabs[1]._weights)

    def test_get_hp_values(self):
        """
        Tests the method get_hp_values().
        """
        hp_values = self.mabs.get_hp_values(
            actions=np.array([
                [2, 1],
                [0, 0],
                [1, 2],
                [3, 2],
                [3, 0],
            ])
        )
        np.testing.assert_equal(np.array([
            ["c3", "d2"],
            ["c1", "d1"],
            ["c2", "d3"],
            ["c4", "d3"],
            ["c4", "d1"],
        ]), hp_values)

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        # Compute the index of the next configuration
        indices = self.mabs(self.configurations)
        np.testing.assert_almost_equal(np.array([
            [1, 2],
            [2, 0],
            [3, 2],
            [0, 1],
            [3, 1],
        ]), indices)

    def test_len(self):
        """
        Tests the magic method __len__().
        """
        self.assertEqual(2, len(self.mabs))

    def test_iter(self):
        """
        Tests the magic method __iter__().
        """
        for mab in self.mabs:
            self.assertIsInstance(mab, MAB)


class TestMAB(unittest.TestCase):
    """
    Tests  the class MAB
    """

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

    def test_normalize_weights(self):
        """
        Tests the method normalize_weights().
        """
        normalized_weights = self.mab._normalize_weights()
        np.testing.assert_almost_equal(np.array([0.25, 0.25, 0.25, 0.25]), normalized_weights)

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

        # Update the mab
        self.mab.update(model=self.model, X=self.X, Y=self.y)

        np.testing.assert_almost_equal(np.array([0.61864039, 1., 0.84648172, 0.8492465]), self.mab._weights)

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        # Compute the index of the next configuration
        indices = self.mab(self.configurations)
        np.testing.assert_almost_equal(np.array([
            [2],
            [2],
            [2],
            [2],
            [1],
        ]), indices)


if __name__ == "__main__":
    unittest.main()

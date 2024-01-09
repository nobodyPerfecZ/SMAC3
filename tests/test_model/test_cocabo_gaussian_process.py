import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.model.gaussian_process import CoCaBOGaussianProcess
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array


# --------------------------------------------------------------
# Test CoCaBOGaussianProcess
# --------------------------------------------------------------
class TestCoCaBOGaussianProcess(unittest.TestCase):

    def setUp(self):
        self.weight = 0.5
        self.kernel = CoCaBOKernel(k1=SimilarityKernel(), k2=RBFKernel(), weight=self.weight)
        self.cs = ConfigurationSpace(
            space={
                "A": (0.1, 1.5),
                "B": (2, 10),
                "C": ["c1", "c2", "c3", "c4"],
                "D": ["d1", "d2", "d3"],
            },
            seed=0,
        )
        configurations = self.cs.sample_configuration(5)
        self.X = convert_configurations_to_array(configurations)
        self.y = np.array([1, 1, 2, 2, 4])
        self.model = CoCaBOGaussianProcess(
            configspace=self.cs,
            kernel=self.kernel,
            n_restarts=20,
            seed=0,
        )

    def test_train(self):
        """
        Tests the method train().
        """
        self.model.train(self.X, self.y)
        np.testing.assert_almost_equal(np.array(
            [1.0, 0.2683963, 11.51292546]
        ), self.model._kernel.theta)


if __name__ == "__main__":
    unittest.main()

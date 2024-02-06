import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.model.gaussian_process import CoCaBOGaussianProcess
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array


class TestCoCaBOGaussianProcess(unittest.TestCase):
    """
    Tests the class CoCaBOGaussianProcess.
    """

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

    def test_predict_with_no_covariance(self):
        """
        Tests the method predict() with covariance_type=None.
        """
        self.model.train(self.X, self.y)
        y_pred, cov = self.model.predict(self.X, covariance_type=None)
        np.testing.assert_almost_equal(np.array([
            [1.0],
            [1.0],
            [2.0],
            [2.0],
            [4.0]
        ]), y_pred)
        self.assertIsNone(cov)

    def test_predict_with_std(self):
        """
        Tests the method predict() with covariance_type=std.
        """
        self.model.train(self.X, self.y)
        y_pred, std = self.model.predict(self.X, covariance_type="std")
        np.testing.assert_almost_equal(np.array([
            [1.0],
            [1.0],
            [2.0],
            [2.0],
            [4.0]
        ]), y_pred)
        np.testing.assert_almost_equal(np.array([
            [1.09544512e-05],
            [1.09544512e-05],
            [1.09544512e-05],
            [1.09544512e-05],
            [1.09544512e-05],
        ]), std)

    def test_predict_with_diagonal_covariance(self):
        """
        Tests the method predict() with covariance_type=diagonal.
        """
        self.model.train(self.X, self.y)
        y_pred, cov = self.model.predict(self.X, covariance_type="diagonal")
        np.testing.assert_almost_equal(np.array([
            [1.0],
            [1.0],
            [2.0],
            [2.0],
            [4.0]
        ]), y_pred)
        np.testing.assert_almost_equal(np.array([
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
        ]), cov)

    def test_predict_with_full_covariance(self):
        """
        Tests the method predict() with covariance_type=full.
        """
        self.model.train(self.X, self.y)
        y_pred, cov = self.model.predict(self.X, covariance_type="full")
        np.testing.assert_almost_equal(np.array([
            [1.0],
            [1.0],
            [2.0],
            [2.0],
            [4.0],
        ]), y_pred)
        np.testing.assert_almost_equal(np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]), cov)


if __name__ == "__main__":
    unittest.main()

import unittest

from ConfigSpace import ConfigurationSpace
from smac.model.gaussian_process.kernels import CoCaBOKernel, SimilarityKernel, RBFKernel
from smac.utils.configspace import convert_configurations_to_array
import numpy as np


# --------------------------------------------------------------
# Test CoCaBOKernel
# --------------------------------------------------------------
class TestCoCaBOKernel(unittest.TestCase):

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
    
    def test_separate_X(self):
        """
        Tests the method separate_X().
        """
        X_categorical, X_continuous = self.kernel._separate_X(self.X)
        np.testing.assert_array_almost_equal(self.X[:, 2:], X_categorical)
        np.testing.assert_array_almost_equal(self.X[:, :2], X_continuous)
    
    def test_diag(self):
        """
        Tests the method diag().
        """
        diag = self.kernel.diag(self.X)
        diag_expected = 1.5 * np.ones(self.X.shape[0])
        np.testing.assert_almost_equal(diag_expected, diag)
    
    def test_call(self):
        """
        Tests the magic method __call__() (without gradient computation).
        """
        K1 = self.kernel(self.X, eval_gradient=False)
        K2 = self.kernel(self.X, self.X, eval_gradient=False)
        K_expected = np.array([
            [1.5, 0.97165073, 0.47229179, 0.95946395, 0.4839997],
            [0.97165073, 1.5, 0.888698, 0.42232887, 0.47919705],
            [0.47229179, 0.888698, 1.5, 0.99874476, 0.88251984],
            [0.95946395, 0.42232887, 0.99874476, 1.5, 0.88804153],
            [0.4839997, 0.47919705, 0.88251984, 0.88804153, 1.5],
        ])
        np.testing.assert_almost_equal(K_expected, K1)
        np.testing.assert_almost_equal(K1, K2)
    
    def test_call_with_grad(self):
        """
        Tests the magic method __call__() (with gradient computation).
        """
        K, K_grad = self.kernel(self.X, eval_gradient=True)
        K_grad_expected = np.array([
            [
                [-1., 1., 0.],
                [-0.98110049, 0.49055024, 0.05561319],
                [-0.94458359, 0., 0.05385175],
                [-0.97297596, 0.48648798, 0.07884064],
                [-0.96799939, 0., 0.03148304],
            ], [
                [-0.98110049, 0.49055024, 0.05561319],
                [-1., 1., 0.],
                [-0.92579867, 0.46289933, 0.20520278],
                [-0.84465774, 0., 0.14259831],
                [-0.95839411, 0., 0.04072811],
            ], [
                [-0.94458359, 0., 0.05385175],
                [-0.92579867, 0.46289933, 0.20520278],
                [-1., 1., 0.],
                [-0.99916317, 0.49958159, 0.00250838],
                [-0.9216799, 0.46083995, 0.2155142 ],
            ], [
                [-0.97297596, 0.48648798, 0.07884064],
                [-0.84465774, 0., 0.14259831],
                [-0.99916317, 0.49958159, 0.00250838],
                [-1., 1., 0.],
                [-0.92536102, 0.46268051, 0.20630414],
            ], [
                [-0.96799939, 0., 0.03148304],
                [-0.95839411, 0., 0.04072811],
                [-0.9216799, 0.46083995, 0.2155142 ],
                [-0.92536102, 0.46268051, 0.20630414],
                [-1., 1., 0.],
            ]
        ])
        K_expected = np.array([
            [1.5, 0.97165073, 0.47229179, 0.95946395, 0.4839997],
            [0.97165073, 1.5, 0.888698, 0.42232887, 0.47919705],
            [0.47229179, 0.888698, 1.5, 0.99874476, 0.88251984],
            [0.95946395, 0.42232887, 0.99874476, 1.5, 0.88804153],
            [0.4839997, 0.47919705, 0.88251984, 0.88804153, 1.5],
        ])
        np.testing.assert_almost_equal(K_expected, K)
        np.testing.assert_almost_equal(K_grad_expected, K_grad)

if __name__ == "__main__":
    unittest.main()
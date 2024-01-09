import unittest

from ConfigSpace import ConfigurationSpace
from smac.model.gaussian_process.kernels import SimilarityKernel
from smac.utils.configspace import convert_configurations_to_array
import numpy as np


# --------------------------------------------------------------
# Test SimilarityKernel
# --------------------------------------------------------------
class TestSimilarityKernel(unittest.TestCase):

    def setUp(self):
        self.noise_level = 1.0
        self.kernel = SimilarityKernel(noise_level=self.noise_level)
        self.cs = ConfigurationSpace(
            space={
                "C": ["c1", "c2", "c3", "c4"],
                "D": ["d1", "d2", "d3"],
            },
            seed=0,
        )
        configurations = self.cs.sample_configuration(5)
        self.X = convert_configurations_to_array(configurations)
        
    def test_diag(self):
        """
        Tests the method diag().
        """
        diag = self.kernel.diag(self.X)
        diag_expected = self.noise_level * np.ones(self.X.shape[0])
        np.testing.assert_almost_equal(diag_expected, diag)
    
    def test_call(self):
        """
        Tests the magic method __call__() (without gradient computation).
        """
        K1 = self.kernel(self.X, eval_gradient=False)
        K2 = self.kernel(self.X, self.X, eval_gradient=False)
        K_expected = self.noise_level * np.array([
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_almost_equal(K_expected, K1)
        np.testing.assert_almost_equal(K1, K2)
    
    def test_call_with_grad(self):
        """
        Tests the magic method __call__() (with gradient computation).
        """
        K, K_grad = self.kernel(self.X, eval_gradient=True)
        K_grad_expected = np.array([
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 1.0],
        ])[:, :, np.newaxis]
        K_expected = self.noise_level * np.array([
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_almost_equal(K_expected, K)
        np.testing.assert_almost_equal(K_grad_expected, K_grad)

if __name__ == "__main__":
    unittest.main()
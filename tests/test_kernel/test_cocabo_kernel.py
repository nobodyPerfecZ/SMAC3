import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.model.gaussian_process.kernels import (
    CoCaBOKernel,
    SimilarityKernel,
    RBFKernel
)
from smac.utils.configspace import convert_configurations_to_array


class TestCoCaBOKernel(unittest.TestCase):
    """
    Tests the class CoCaBOKernel.
    """

    def setUp(self):
        self.weight = np.exp(0.5)
        self.weight_bounds = (1, np.exp(1))

        self.noise_level = 1.0
        self.noise_level_bounds = (1e-5, 1e5)

        self.length_scale = 1.0
        self.length_scale_bounds = (1e-5, 1e5)

        self.kernel = CoCaBOKernel(
            k1=SimilarityKernel(noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds),
            k2=RBFKernel(length_scale=self.length_scale, length_scale_bounds=self.length_scale_bounds),
            weight=self.weight,
            weight_bounds=self.weight_bounds,
        )

        self.mixed_cs = ConfigurationSpace(
            space={
                "A": (0.1, 1.5),
                "B": (2, 10),
                "C": ["c1", "c2", "c3", "c4"],
                "D": ["d1", "d2", "d3"],
            },
            seed=0,
        )
        self.mixed_X = convert_configurations_to_array(self.mixed_cs.sample_configuration(5))

        self.categorical_cs = ConfigurationSpace(
            space={
                "C": ["c1", "c2", "c3", "c4"],
                "D": ["d1", "d2", "d3"],
            },
            seed=0,
        )
        self.categorical_X = convert_configurations_to_array(self.categorical_cs.sample_configuration(5))

        self.continuous_cs = ConfigurationSpace(
            space={
                "A": (0.1, 1.5),
                "B": (2, 10),
            },
            seed=0,
        )
        self.continuous_X = convert_configurations_to_array(self.continuous_cs.sample_configuration(5))

    def test_hyperparameter_weight(self):
        """
        Tests the property hyperparameter_weight.
        """
        self.assertEqual("weight", self.kernel.hyperparameter_weight.name)
        np.testing.assert_almost_equal(np.array([[1, np.e]]), self.kernel.hyperparameter_weight.bounds)
        self.assertEqual("numeric", self.kernel.hyperparameter_weight.value_type)

    def test_hyperparameters(self):
        """
        Tests the property hyperparameters.
        """
        # Extract the hyperparameters of the CoCaBO kernel
        cocabo_hp, k1_hp, k2_hp = self.kernel.hyperparameters

        self.assertEqual("weight", cocabo_hp.name)
        np.testing.assert_almost_equal(np.array([[1, np.e]]), cocabo_hp.bounds)
        self.assertEqual("numeric", cocabo_hp.value_type)

        self.assertEqual("k1__noise_level", k1_hp.name)
        np.testing.assert_almost_equal(np.array([[1e-05, 1e05]]), k1_hp.bounds)
        self.assertEqual("numeric", k1_hp.value_type)

        self.assertEqual("k2__length_scale", k2_hp.name)
        np.testing.assert_almost_equal(np.array([[1e-05, 1e05]]), k2_hp.bounds)
        self.assertEqual("numeric", k2_hp.value_type)

    def test_theta(self):
        """
        Tests the property theta.
        """
        thetas = np.log(np.array([self.weight, self.noise_level, self.length_scale]))
        np.testing.assert_almost_equal(thetas, self.kernel.theta)

        new_thetas = np.array([1.0, 1.0, 1.0])
        self.kernel.theta = new_thetas
        np.testing.assert_almost_equal(new_thetas, self.kernel.theta)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        np.testing.assert_almost_equal(np.array([
            np.log(np.array(self.weight_bounds)),
            np.log(np.array(self.noise_level_bounds)),
            np.log(np.array(self.length_scale_bounds)),
        ]), self.kernel.bounds)

    def test_separate_X_with_mixed_feature_space(self):
        """
        Tests the method separate_X() with mixed feature space (categorical + continuous).
        """
        X_categorical, X_continuous = self.kernel._separate_X(self.mixed_X)
        np.testing.assert_array_almost_equal(self.mixed_X[:, 2:], X_categorical)
        np.testing.assert_array_almost_equal(self.mixed_X[:, :2], X_continuous)

    def test_separate_X_with_categorical_feature_space(self):
        """
        Tests the method separate_X() with a feature space that contains only categorical features.
        """
        X_categorical, X_continuous = self.kernel._separate_X(self.categorical_X)
        np.testing.assert_array_almost_equal(self.categorical_X, X_categorical)
        self.assertEqual(0, X_continuous.size)

    def test_separate_X_with_continuous_feature_space(self):
        """
        Tests the method separate_X() with a feature space that contains only continuous features.
        """
        X_categorical, X_continuous = self.kernel._separate_X(self.continuous_X)
        self.assertEqual(0, X_categorical.size)
        np.testing.assert_array_almost_equal(self.continuous_X, X_continuous)

    def test_get_params(self):
        """
        Tests the method get_params().
        """
        params_dict = self.kernel.get_params()

        self.assertEqual(self.noise_level, params_dict["k1__noise_level"])
        self.assertEqual(self.noise_level_bounds, params_dict["k1__noise_level_bounds"])
        self.assertEqual(self.length_scale, params_dict["k2__length_scale"])
        self.assertEqual(self.length_scale_bounds, params_dict["k2__length_scale_bounds"])
        self.assertEqual(self.weight, params_dict["weight"])
        self.assertEqual(self.weight_bounds, params_dict["weight_bounds"])

    def test_diag_with_mixed_feature_space(self):
        """
        Tests the method diag() with mixed feature space (categorical + continuous).
        """
        diag = self.kernel.diag(self.mixed_X)
        diag_expected = 1.5 * np.ones(self.mixed_X.shape[0])
        np.testing.assert_almost_equal(diag_expected, diag)

    def test_diag_with_categorical_feature_space(self):
        """
        Tests the method diag() with a feature space that contains only categorical features.
        """
        diag = self.kernel.diag(self.categorical_X)
        diag_expected = np.ones(self.categorical_X.shape[0])
        np.testing.assert_almost_equal(diag_expected, diag)

    def test_diag_with_continuous_feature_space(self):
        """
        Tests the method diag() with a feature space that contains only continuous features.
        """
        diag = self.kernel.diag(self.continuous_X)
        diag_expected = np.ones(self.continuous_X.shape[0])
        np.testing.assert_almost_equal(diag_expected, diag)

    def test_call_with_mixed_feature_space(self):
        """
        Tests the magic method __call__() without gradient computation and with a mixed feature space
        (categorical + continuous).
        """
        K1 = self.kernel(self.mixed_X, eval_gradient=False)
        K2 = self.kernel(self.mixed_X, self.mixed_X, eval_gradient=False)
        K_expected = np.array([
            [1.5, 0.97165073, 0.47229179, 0.95946395, 0.4839997],
            [0.97165073, 1.5, 0.888698, 0.42232887, 0.47919705],
            [0.47229179, 0.888698, 1.5, 0.99874476, 0.88251984],
            [0.95946395, 0.42232887, 0.99874476, 1.5, 0.88804153],
            [0.4839997, 0.47919705, 0.88251984, 0.88804153, 1.5],
        ])
        np.testing.assert_almost_equal(K_expected, K1)
        np.testing.assert_almost_equal(K1, K2)
    
    def test_call_with_categorical_feature_space(self):
        """
        Tests the magic method __call__() without gradient computation and with a feature space that contains only
        categorical features.
        """
        K1 = self.kernel(self.categorical_X, eval_gradient=False)
        K2 = self.kernel(self.categorical_X, self.categorical_X, eval_gradient=False)
        K_expected = np.array([
            [1., 1., 0.5, 0.5, 0.5],
            [1., 1., 0.5, 0.5, 0.5],
            [0.5, 0.5, 1., 1., 0.],
            [0.5, 0.5, 1., 1., 0.],
            [0.5, 0.5, 0., 0., 1.],
        ])
        np.testing.assert_almost_equal(K_expected, K1)
        np.testing.assert_almost_equal(K1, K2)

    def test_call_with_continuous_feature_space(self):
        """
        Tests the magic method __call__() without gradient computation and with a feature space that contains only
        continuous features.
        """
        K1 = self.kernel(self.continuous_X, eval_gradient=False)
        K2 = self.kernel(self.continuous_X, self.continuous_X, eval_gradient=False)
        K_expected = np.array([
            [1., 0.96220098, 0.94458359, 0.94595193, 0.96799939],
            [0.96220098, 1., 0.85159734, 0.84465774, 0.95839411],
            [0.94458359, 0.85159734, 1., 0.99832634, 0.84335979],
            [0.94595193, 0.84465774, 0.99832634, 1., 0.85072204],
            [0.96799939, 0.95839411, 0.84335979, 0.85072204, 1.],
        ])
        np.testing.assert_almost_equal(K_expected, K1)
        np.testing.assert_almost_equal(K1, K2)

    def test_call_with_grad_and_mixed_feature_space(self):
        """
        Tests the magic method __call__() with gradient computation and with a mixed feature space
        (categorical + continuous).
        """
        K, K_grad = self.kernel(self.mixed_X, eval_gradient=True)
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
                [-0.9216799, 0.46083995, 0.2155142],
            ], [
                [-0.97297596, 0.48648798, 0.07884064],
                [-0.84465774, 0., 0.14259831],
                [-0.99916317, 0.49958159, 0.00250838],
                [-1., 1., 0.],
                [-0.92536102, 0.46268051, 0.20630414],
            ], [
                [-0.96799939, 0., 0.03148304],
                [-0.95839411, 0., 0.04072811],
                [-0.9216799, 0.46083995, 0.2155142],
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

    def test_call_with_grad_and_categorical_feature_space(self):
        """
        Tests the magic method __call__() with gradient computation and with a feature space that contains only
        categorical features.
        """
        K, K_grad = self.kernel(self.categorical_X, eval_gradient=True)
        K_grad_expected = np.array([
            [
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0.5, 0.],
                [0., 0.5, 0.],
                [0., 0.5, 0.],
            ], [
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0.5, 0.],
                [0., 0.5, 0.],
                [0., 0.5, 0.],
            ], [
                [0., 0.5, 0.],
                [0., 0.5, 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
            ], [
                [0., 0.5, 0.],
                [0., 0.5, 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
            ], [
                [0., 0.5, 0.],
                [0., 0.5, 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
            ]
        ])
        K_expected = np.array([
            [1., 1., 0.5, 0.5, 0.5],
            [1., 1., 0.5, 0.5, 0.5],
            [0.5, 0.5, 1., 1., 0.],
            [0.5, 0.5, 1., 1., 0.],
            [0.5, 0.5, 0., 0., 1.],
        ])
        np.testing.assert_almost_equal(K_expected, K)
        np.testing.assert_almost_equal(K_grad_expected, K_grad)

    def test_call_with_grad_and_continuous_feature_space(self):
        """
        Tests the magic method __call__() with gradient computation and with a feature space that contains only
        continuous features.
        """
        K, K_grad = self.kernel(self.continuous_X, eval_gradient=True)
        K_grad_expected = np.array([
            [
                [0., 0., 0.],
                [0., 0., 0.07415092],
                [0., 0., 0.10770349],
                [0., 0., 0.10512085],
                [0., 0., 0.06296607],
            ], [
                [0., 0., 0.07415092],
                [0., 0., 0.],
                [0., 0., 0.2736037],
                [0., 0., 0.28519661],
                [0., 0., 0.08145622],
            ], [
                [0., 0., 0.10770349],
                [0., 0., 0.2736037],
                [0., 0., 0.],
                [0., 0., 0.00334451],
                [0., 0., 0.28735227],
            ], [
                [0., 0., 0.10512085],
                [0., 0., 0.28519661],
                [0., 0., 0.00334451],
                [0., 0., 0.],
                [0., 0., 0.27507218],
            ], [
                [0., 0., 0.06296607],
                [0., 0., 0.08145622],
                [0., 0., 0.28735227],
                [0., 0., 0.27507218],
                [0., 0., 0.],
            ]
        ])
        K_expected = np.array([
            [1., 0.96220098, 0.94458359, 0.94595193, 0.96799939],
            [0.96220098, 1., 0.85159734, 0.84465774, 0.95839411],
            [0.94458359, 0.85159734, 1., 0.99832634, 0.84335979],
            [0.94595193, 0.84465774, 0.99832634, 1., 0.85072204],
            [0.96799939, 0.95839411, 0.84335979, 0.85072204, 1.],
        ])
        np.testing.assert_almost_equal(K_expected, K)
        np.testing.assert_almost_equal(K_grad_expected, K_grad)


if __name__ == "__main__":
    unittest.main()

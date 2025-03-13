"""
Unit tests for the SplineFitter class.

These tests verify that the SplineFitter class correctly replicates
the behavior of R's smooth.spline with cv=TRUE.
"""

import numpy as np
from scipy import interpolate  # type: ignore

from thermogram_baseline.spline_fitter import SplineFitter


class TestSplineFitter:
    """Tests for the SplineFitter class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.fitter = SplineFitter()

        # Create standard test data for consistent testing
        # Simple sine wave with noise
        self.x_simple = np.linspace(0, 10, 100)
        self.y_simple = np.sin(self.x_simple) + 0.1 * np.random.randn(100)

        # More complex curve resembling a thermogram
        temp_range = np.linspace(45, 90, 450)
        self.x_thermo = temp_range
        # Create a curve with multiple peaks
        peak1 = 0.3 * np.exp(-0.5 * ((temp_range - 63) / 2) ** 2)  # Peak ~ 63°C
        peak2 = 0.2 * np.exp(-0.5 * ((temp_range - 70) / 2) ** 2)  # Peak ~ 70°C
        peak3 = 0.15 * np.exp(-0.5 * ((temp_range - 77) / 2.5) ** 2)  # Peak ~ 77°C
        baseline = 0.02 * (temp_range - 65)  # Slight linear baseline
        self.y_thermo = (peak1 + peak2 + peak3 + baseline) + 0.02 * np.random.randn(
            len(temp_range)
        )

    def test_splinefitter_initialization(self) -> None:
        """Test that the SplineFitter can be initialized."""
        assert isinstance(self.fitter, SplineFitter)

    def test_fit_with_gcv_returns_spline(self) -> None:
        """Test that fit_with_gcv returns a UnivariateSpline object."""
        spline = self.fitter.fit_with_gcv(self.x_simple, self.y_simple)
        assert isinstance(spline, interpolate.UnivariateSpline)

    def test_fit_with_gcv_handles_unsorted_data(self) -> None:
        """Test that fit_with_gcv can handle unsorted data."""
        # Randomly shuffle the data
        idx = np.random.permutation(len(self.x_simple))
        x_unsorted = self.x_simple[idx]
        y_unsorted = self.y_simple[idx]

        # Fit should work without error
        spline = self.fitter.fit_with_gcv(x_unsorted, y_unsorted)
        assert isinstance(spline, interpolate.UnivariateSpline)

        # Evaluate at sorted points to check (use np.argsort for stable sorting)
        sorted_idx = np.argsort(x_unsorted)
        sorted_x = x_unsorted[sorted_idx]
        y_fit = spline(sorted_x)
        assert len(y_fit) == len(x_unsorted)

    def test_fit_with_gcv_handles_repeated_x_values(self) -> None:
        """Test that fit_with_gcv can handle repeated x values."""
        # Create data with repeated x values
        x_repeated = np.concatenate([self.x_simple, self.x_simple[0:5]])
        y_repeated = np.concatenate([self.y_simple, self.y_simple[0:5]])

        # Fit should work without error by averaging y values for repeated x
        spline = self.fitter.fit_with_gcv(x_repeated, y_repeated)
        assert isinstance(spline, interpolate.UnivariateSpline)

    def test_fit_quality_on_simple_data(self) -> None:
        """Test that the fitted spline matches the data well."""
        # Fit the spline
        spline = self.fitter.fit_with_gcv(self.x_simple, self.y_simple)

        # Predict values
        y_fit = spline(self.x_simple)

        # Calculate R² to measure fit quality
        ss_total = np.sum((self.y_simple - np.mean(self.y_simple)) ** 2)
        ss_residual = np.sum((self.y_simple - y_fit) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # For a good fit, R² should be reasonably high
        assert r_squared > 0.8

    def test_fit_has_appropriate_smoothing(self) -> None:
        """Test that the fitted spline has appropriate smoothing."""
        # Fit the spline
        spline = self.fitter.fit_with_gcv(self.x_thermo, self.y_thermo)

        # Calculate first and second derivatives
        x_eval = np.linspace(self.x_thermo.min(), self.x_thermo.max(), 200)
        y_eval = spline(x_eval)

        # Calculate approximate second derivatives
        h = x_eval[1] - x_eval[0]
        second_deriv = np.diff(np.diff(y_eval) / h) / h

        # For appropriate smoothing, the second derivative shouldn't be too large
        assert np.max(np.abs(second_deriv)) < 10

    def test_spline_handles_edge_cases(self) -> None:
        """Test that the spline fitter handles edge cases appropriately."""
        # Test with very few points
        x_few = np.array([1, 2, 3, 4])
        y_few = np.array([1, 4, 9, 16])
        spline_few = self.fitter.fit_with_gcv(x_few, y_few)
        assert isinstance(spline_few, interpolate.UnivariateSpline)

        # Test with constant y values
        x_const = np.linspace(0, 10, 20)
        y_const = np.ones_like(x_const)
        spline_const = self.fitter.fit_with_gcv(x_const, y_const)
        assert isinstance(spline_const, interpolate.UnivariateSpline)
        y_pred = spline_const(x_const)
        assert np.allclose(y_pred, y_const, atol=0.1)

    def test_smoothing_parameter_is_reasonable(self) -> None:
        """Test that the selected smoothing parameter is reasonable."""
        # Fit the spline
        spline = self.fitter.fit_with_gcv(self.x_thermo, self.y_thermo)

        # Check that smoothing parameter is not too small or too large
        s = getattr(spline, "s_opt", None)
        assert s is not None
        assert s > 0, "Smoothing parameter should be positive"
        assert s < len(self.x_thermo) * 100, (
            "Smoothing parameter shouldn't be unreasonably large"
        )

        # For UnivariateSpline, we can also check the number of knots
        # More knots = less smoothing
        n_knots = len(spline.get_knots())
        assert n_knots < len(self.x_thermo), (
            "Number of knots should be less than number of data points"
        )
        assert n_knots > 5, (
            "Number of knots should be reasonable for capturing main features"
        )


# Add mock R comparison tests that will be updated once we have real R output
def test_compare_with_r_output_mock() -> None:
    """
    Test comparison with mock R output.

    Note: This is a placeholder test. In the real implementation,
    we would compare against actual R output.
    """
    fitter = SplineFitter()
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)

    # Mock R output (using the same function for now)
    mock_r_spline = interpolate.UnivariateSpline(x, y, s=len(x) * 0.1)
    mock_r_fitted = mock_r_spline(x)

    # Compare outputs
    comparison = fitter.compare_with_r_output(x, y, mock_r_fitted)

    # Check keys in comparison dictionary
    assert "mean_squared_error" in comparison
    assert "mean_absolute_error" in comparison

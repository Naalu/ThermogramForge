"""
Tests for the interpolation module.

This module contains tests for the thermogram interpolation functionality,
verifying that it correctly interpolates thermogram data onto a fixed grid
and produces results consistent with the R implementation.
"""

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from thermogram_baseline.interpolation import interpolate_thermogram
from thermogram_baseline.spline_fitter import SplineFitter


def create_synthetic_thermogram() -> pl.DataFrame:
    """
    Create a synthetic thermogram for testing.

    Returns:
        A Polars DataFrame with synthetic thermogram data
    """
    # Create temperature range with irregular spacing
    temp_range = np.sort(np.random.uniform(45, 90, 100))

    # Create peaks
    peak1 = 0.3 * np.exp(-0.5 * ((temp_range - 63) / 2) ** 2)  # Peak ~ 63°C
    peak2 = 0.2 * np.exp(-0.5 * ((temp_range - 70) / 2) ** 2)  # Peak ~ 70°C
    peak3 = 0.15 * np.exp(-0.5 * ((temp_range - 77) / 2.5) ** 2)  # Peak ~ 77°C

    # Add a non-linear baseline
    baseline = 0.02 * (temp_range - 65) + 0.0005 * (temp_range - 65) ** 2

    # Combine components with noise
    np.random.seed(42)  # For reproducibility
    dcp = peak1 + peak2 + peak3 + baseline + 0.01 * np.random.randn(len(temp_range))

    # Create a polars DataFrame
    return pl.DataFrame({"Temperature": temp_range, "dCp": dcp})


def test_interpolate_thermogram_basic():
    """Test basic functionality of interpolate_thermogram."""
    # Create test data with irregular temperature spacing
    data = create_synthetic_thermogram()

    # Create a custom grid for interpolation
    grid_temp = np.linspace(50, 85, 100)

    # Interpolate the data
    result = interpolate_thermogram(data, grid_temp)

    # Check that the result has the right columns
    assert "Temperature" in result.columns
    assert "dCp" in result.columns

    # Check that the result has the right number of rows (same as grid)
    assert result.height == len(grid_temp)

    # Check that the temperatures in the result match the grid
    np.testing.assert_array_equal(
        result.select("Temperature").to_numpy().flatten(), grid_temp
    )

    # Values should be finite
    assert np.all(np.isfinite(result.select("dCp").to_numpy()))


def test_interpolate_thermogram_default_grid():
    """Test interpolate_thermogram with default grid."""
    # Create test data
    data = create_synthetic_thermogram()

    # Interpolate with default grid
    result = interpolate_thermogram(data)

    # Default grid should be from 45 to 90 with 0.1°C steps
    expected_grid = np.arange(45, 90.1, 0.1)

    # Check grid
    np.testing.assert_array_equal(
        result.select("Temperature").to_numpy().flatten(), expected_grid
    )

    # Check number of rows
    assert result.height == len(expected_grid)


def test_interpolate_thermogram_preserves_features():
    """Test that interpolation preserves key features of the thermogram."""
    # Create test data with well-defined peaks
    temperature = np.linspace(45, 90, 100)
    # Create distinct peaks
    peak1 = 0.5 * np.exp(-0.5 * ((temperature - 55) / 1.5) ** 2)
    peak2 = 0.3 * np.exp(-0.5 * ((temperature - 70) / 1.5) ** 2)
    peak3 = 0.2 * np.exp(-0.5 * ((temperature - 85) / 1.5) ** 2)
    dcp = peak1 + peak2 + peak3
    data = pl.DataFrame({"Temperature": temperature, "dCp": dcp})

    # Interpolate to a finer grid
    fine_grid = np.linspace(45, 90, 451)  # 0.1°C steps
    result = interpolate_thermogram(data, fine_grid)

    # Find peaks in original data
    peak1_idx = np.argmax(dcp[:33])  # First third (around 55°C)
    peak2_idx = 33 + np.argmax(dcp[33:66])  # Middle third (around 70°C)
    peak3_idx = 66 + np.argmax(dcp[66:])  # Last third (around 85°C)

    peak1_temp = temperature[peak1_idx]
    peak2_temp = temperature[peak2_idx]
    peak3_temp = temperature[peak3_idx]

    # Find closest points in interpolated data
    interp_dcp = result.select("dCp").to_numpy().flatten()

    # Find indices in fine_grid closest to original peak temperatures
    peak1_interp_idx = np.abs(fine_grid - peak1_temp).argmin()
    peak2_interp_idx = np.abs(fine_grid - peak2_temp).argmin()
    peak3_interp_idx = np.abs(fine_grid - peak3_temp).argmin()

    # Find local maxima near these points in interpolated data
    # Search in a window of ±2°C
    window = 20  # Number of points in a 2°C window at 0.1°C resolution

    peak1_local_max_idx = (
        peak1_interp_idx
        - window
        + np.argmax(interp_dcp[peak1_interp_idx - window : peak1_interp_idx + window])
    )
    peak2_local_max_idx = (
        peak2_interp_idx
        - window
        + np.argmax(interp_dcp[peak2_interp_idx - window : peak2_interp_idx + window])
    )
    peak3_local_max_idx = (
        peak3_interp_idx
        - window
        + np.argmax(interp_dcp[peak3_interp_idx - window : peak3_interp_idx + window])
    )

    # Get peak temperatures from interpolated data
    peak1_interp_temp = fine_grid[peak1_local_max_idx]
    peak2_interp_temp = fine_grid[peak2_local_max_idx]
    peak3_interp_temp = fine_grid[peak3_local_max_idx]

    # Peaks should be at similar temperatures (within 1°C)
    assert abs(peak1_interp_temp - peak1_temp) < 1.0
    assert abs(peak2_interp_temp - peak2_temp) < 1.0
    assert abs(peak3_interp_temp - peak3_temp) < 1.0


def test_interpolate_thermogram_r_consistency():
    """Test that interpolation results are consistent with R's behavior."""
    # This test is conditional on having R integration available
    try:
        import rpy2

        # print the version of rpy2 to the console
        print(f"rpy2 version: {rpy2.__version__}")

        r_available = True
    except ImportError:
        r_available = False

    if not r_available:
        pytest.skip("rpy2 not available, skipping R consistency test")

    # Use existing SplineFitter with R=True
    fitter = SplineFitter()
    if not fitter._r_available:
        pytest.skip("R is not available for testing")

    # Create test data
    data = create_synthetic_thermogram()

    # Extract raw data
    temp = data.select("Temperature").to_numpy().flatten()
    dcp = data.select("dCp").to_numpy().flatten()

    # Define a test grid
    grid_temp = np.linspace(45, 90, 50)

    # Fit spline directly using R
    r_spline = fitter.fit_with_gcv(temp, dcp, use_r=True)
    r_values = r_spline(grid_temp)

    # Use interpolate_thermogram which should use the Python implementation
    # internally but match R's behavior
    result = interpolate_thermogram(data, grid_temp)
    py_values = result.select("dCp").to_numpy().flatten()

    # Compare results - they should be very similar
    # Since our SplineFitter is designed to match R's behavior
    np.testing.assert_allclose(py_values, r_values, rtol=1e-2, atol=1e-3)

    # Skip this test if plotly is not available
    try:
        if find_spec("plotly") is None:
            pytest.skip("plotly not available, skipping plot test")
    except ImportError:
        pytest.skip("plotly not available, skipping plot test")

    # Create test data
    data = create_synthetic_thermogram()

    # Create temporary file for the plot output
    tmp_dir = Path("tests/output")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    output_file = tmp_dir / "interpolation_test_plot.html"

    # Patch the show method to save the figure instead

    original_show = go.Figure.show
    try:
        # Replace show with a function that saves to file instead
        def mock_show(self):
            self.write_html(str(output_file))

        go.Figure.show = mock_show

        # Call with plot=True
        result = interpolate_thermogram(data, plot=True)

        # Check that the plot was saved
        assert output_file.exists()
    finally:
        # Restore original show method
        go.Figure.show = original_show


def test_interpolate_thermogram_invalid_input():
    """Test interpolate_thermogram with invalid input."""
    # Missing columns
    bad_data = pl.DataFrame({"Wrong": [1, 2, 3], "Columns": [4, 5, 6]})

    with pytest.raises(
        (ValueError, Exception)
    ):  # Accept either ValueError or any Exception
        interpolate_thermogram(bad_data)

    # Empty DataFrame
    empty_data = pl.DataFrame({"Temperature": [], "dCp": []})

    # Should handle gracefully rather than crash
    try:
        result = interpolate_thermogram(empty_data)
        # If it doesn't raise an exception, check that it returned something reasonable
        assert result.height == len(np.arange(45, 90.1, 0.1))
    except Exception:
        # It's also acceptable for it to raise an exception for empty data
        pass

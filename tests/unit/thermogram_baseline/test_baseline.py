"""Tests for the main baseline processing workflow."""

import numpy as np

from packages.thermogram_baseline.thermogram_baseline.baseline import auto_baseline
from packages.thermogram_baseline.thermogram_baseline.types import (
    InterpolatedResult,
    ThermogramData,
)
from tests.test_data_utils import (
    generate_real_like_thermogram,
    generate_simple_thermogram,
    thermogram_to_dataframe,
)


def test_auto_baseline_with_simple_thermogram():
    """Test the complete auto_baseline workflow with a simple thermogram."""
    # Generate a simple test thermogram
    test_data = generate_simple_thermogram(
        baseline_slope=0.01,
        baseline_intercept=0.5,
    )

    # Process the thermogram
    result = auto_baseline(
        test_data,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
        point_selection="innermost",
        verbose=True,
    )

    # Check result structure
    assert isinstance(result, InterpolatedResult)
    assert isinstance(result.data, ThermogramData)
    assert isinstance(result.grid_temp, np.ndarray)
    assert result.original_data is not None
    assert result.baseline_result is not None

    # Check grid properties
    assert len(result.data.temperature) == 451  # Default grid: 45 to 90 by 0.1
    assert result.data.temperature[0] == 45.0
    assert result.data.temperature[-1] == 90.0

    # Check baseline subtraction effect
    # The peak should still be present in the processed data
    peak_idx = np.abs(result.data.temperature - 70.0).argmin()
    assert result.data.dcp[peak_idx] > 0.5

    # Endpoint regions should be close to zero
    lower_idx = np.abs(result.data.temperature - 50.0).argmin()
    upper_idx = np.abs(result.data.temperature - 85.0).argmin()
    assert abs(result.data.dcp[lower_idx]) < 0.2
    assert abs(result.data.dcp[upper_idx]) < 0.2


def test_auto_baseline_with_dataframe_input():
    """Test auto_baseline with DataFrame input."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Process the thermogram
    result = auto_baseline(
        test_df,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
    )

    # Check result structure
    assert isinstance(result, InterpolatedResult)
    assert isinstance(result.data, ThermogramData)

    # Check that processing completed successfully
    assert len(result.data.temperature) == 451


def test_auto_baseline_with_custom_grid():
    """Test auto_baseline with a custom temperature grid."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Define a custom grid
    custom_grid = np.linspace(50, 85, 100)

    # Process the thermogram
    result = auto_baseline(test_data, grid_temp=custom_grid)

    # Check grid properties
    assert len(result.data.temperature) == 100
    assert result.data.temperature[0] == 50.0
    assert result.data.temperature[-1] == 85.0
    assert np.allclose(result.data.temperature, custom_grid)


def test_auto_baseline_with_different_point_selection():
    """Test auto_baseline with different point selection methods."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Test with different point selection methods
    for method in ["innermost", "outmost", "mid"]:
        result = auto_baseline(
            test_data,
            point_selection=method,
        )

        # Check that endpoint method was properly applied
        assert result.baseline_result.endpoints.method == method


def test_auto_baseline_with_complex_thermogram():
    """Test auto_baseline with a more complex, realistic thermogram."""
    # Generate a more realistic test thermogram
    test_data = generate_real_like_thermogram()

    # Process the thermogram
    result = auto_baseline(
        test_data,
        window_size=50,
        exclusion_lower=55.0,
        exclusion_upper=85.0,
        verbose=True,
    )

    # Check that processing completed successfully
    assert isinstance(result, InterpolatedResult)

    # The processed data should preserve the peaks
    # Check if peaks are still present at expected locations
    for peak_temp in [61, 68, 76]:
        peak_idx = np.abs(result.data.temperature - peak_temp).argmin()
        # Each peak should have a positive value in the processed data
        assert result.data.dcp[peak_idx] > 0.2

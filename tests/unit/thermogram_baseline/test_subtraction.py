"""Tests for the baseline subtraction module."""

import numpy as np

from packages.thermogram_baseline.thermogram_baseline.subtraction import (
    subtract_baseline,
)
from packages.thermogram_baseline.thermogram_baseline.types import ThermogramData
from tests.test_data_utils import (
    generate_multi_peak_thermogram,
    generate_simple_thermogram,
    thermogram_to_dataframe,
)


def test_subtract_baseline_with_simple_thermogram():
    """Test baseline subtraction with a simple thermogram."""
    # Generate a simple test thermogram with known baseline
    baseline_slope = 0.01
    baseline_intercept = 0.5
    test_data = generate_simple_thermogram(
        baseline_slope=baseline_slope,
        baseline_intercept=baseline_intercept,
    )

    # Subtract baseline
    result = subtract_baseline(
        test_data,
        lower_temp=50.0,
        upper_temp=85.0,
    )

    # Check result structure
    assert result.original is test_data
    assert isinstance(result.baseline, ThermogramData)
    assert isinstance(result.subtracted, ThermogramData)
    assert result.endpoints.lower == 50.0
    assert result.endpoints.upper == 85.0

    # Check baseline subtraction effect
    # For a simple thermogram, the baseline-subtracted values at the endpoints
    # should be close to zero
    lower_idx = np.abs(test_data.temperature - 50.0).argmin()
    upper_idx = np.abs(test_data.temperature - 85.0).argmin()

    assert abs(result.subtracted.dcp[lower_idx]) < 0.1
    assert abs(result.subtracted.dcp[upper_idx]) < 0.1

    # The peak should still be present in the subtracted data
    peak_idx = np.abs(test_data.temperature - 70.0).argmin()
    assert result.subtracted.dcp[peak_idx] > 0.5


def test_subtract_baseline_with_dataframe_input():
    """Test baseline subtraction with DataFrame input."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Subtract baseline
    result = subtract_baseline(
        test_df,
        lower_temp=50.0,
        upper_temp=85.0,
    )

    # Check result structure
    assert isinstance(result.original, ThermogramData)
    assert isinstance(result.baseline, ThermogramData)
    assert isinstance(result.subtracted, ThermogramData)

    # Check basic properties
    assert len(result.original.temperature) == len(test_df)
    assert len(result.subtracted.temperature) == len(test_df)


def test_subtract_baseline_endpoint_validation():
    """Test endpoint validation in baseline subtraction."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram(
        min_temp=45.0,
        max_temp=90.0,
    )

    # Test endpoints at extremes of data range
    result = subtract_baseline(
        test_data,
        lower_temp=44.0,  # Below min_temp
        upper_temp=91.0,  # Above max_temp
    )

    # Should adjust endpoints to be within acceptable range
    assert result.endpoints.lower > 45.0
    assert result.endpoints.upper < 90.0


def test_subtract_baseline_with_multi_peak_thermogram():
    """Test baseline subtraction with a multi-peak thermogram."""
    # Generate a more complex test thermogram
    test_data = generate_multi_peak_thermogram()

    # Subtract baseline
    result = subtract_baseline(
        test_data,
        lower_temp=50.0,
        upper_temp=85.0,
    )

    # The subtracted data should preserve the peaks
    # Check if peaks are still present at expected locations
    for peak_temp in [60.0, 70.0, 80.0]:
        peak_idx = np.abs(test_data.temperature - peak_temp).argmin()
        # Each peak should have a positive value in the subtracted data
        assert result.subtracted.dcp[peak_idx] > 0.1

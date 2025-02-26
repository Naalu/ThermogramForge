"""Tests for the endpoint detection module."""

import numpy as np
import pytest

from packages.thermogram_baseline.thermogram_baseline.detection import detect_endpoints
from packages.thermogram_baseline.thermogram_baseline.types import ThermogramData
from tests.test_data_utils import (
    generate_multi_peak_thermogram,
    generate_simple_thermogram,
    thermogram_to_dataframe,
)


def test_detect_endpoints_with_simple_thermogram():
    """Test endpoint detection with a simple thermogram."""
    # Generate a simple test thermogram
    test_data = generate_simple_thermogram(
        min_temp=45.0,
        max_temp=90.0,
        peak_center=70.0,
        baseline_slope=0.01,
    )

    # Detect endpoints
    endpoints = detect_endpoints(
        test_data,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
        point_selection="innermost",
    )

    # Check that endpoints are within expected ranges
    assert 45.0 <= endpoints.lower <= 60.0, (
        f"Lower endpoint {endpoints.lower} outside expected range"
    )
    assert 80.0 <= endpoints.upper <= 90.0, (
        f"Upper endpoint {endpoints.upper} outside expected range"
    )
    assert endpoints.method == "innermost"


def test_detect_endpoints_with_dataframe_input():
    """Test endpoint detection with DataFrame input."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Detect endpoints
    endpoints = detect_endpoints(
        test_df,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
    )

    # Check that endpoints are within expected ranges
    assert 45.0 <= endpoints.lower <= 60.0
    assert 80.0 <= endpoints.upper <= 90.0


def test_detect_endpoints_with_different_point_selection():
    """Test endpoint detection with different point selection methods."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Test with different point selection methods
    for method in ["innermost", "outmost", "mid"]:
        endpoints = detect_endpoints(
            test_data,
            window_size=50,
            exclusion_lower=60.0,
            exclusion_upper=80.0,
            point_selection=method,
        )

        # Check that endpoints are within expected ranges
        assert 45.0 <= endpoints.lower <= 60.0
        assert 80.0 <= endpoints.upper <= 90.0
        assert endpoints.method == method


def test_detect_endpoints_input_validation():
    """Test input validation in endpoint detection."""
    # Create a small dataset that will trigger validation errors
    small_data = ThermogramData(
        temperature=np.linspace(50, 85, 20), dcp=np.random.normal(0, 0.1, 20)
    )

    # Test window size too large
    with pytest.raises(ValueError, match="Not enough data points"):
        detect_endpoints(small_data, window_size=30)

    # Make a dataset with enough points for window size but wrong exclusion bounds
    larger_data = ThermogramData(
        temperature=np.linspace(50, 85, 100), dcp=np.random.normal(0, 0.1, 100)
    )

    # Test exclusion bounds outside data range
    with pytest.raises(ValueError, match="Exclusion zone lower bound"):
        detect_endpoints(larger_data, window_size=30, exclusion_lower=40.0)

    with pytest.raises(ValueError, match="Exclusion zone upper bound"):
        detect_endpoints(larger_data, window_size=30, exclusion_upper=90.0)


def test_detect_endpoints_with_multi_peak_thermogram():
    """Test endpoint detection with a multi-peak thermogram."""
    # Generate a more complex test thermogram
    test_data = generate_multi_peak_thermogram()

    # Detect endpoints
    endpoints = detect_endpoints(
        test_data,
        window_size=50,
        exclusion_lower=55.0,
        exclusion_upper=85.0,
    )

    # Check that endpoints are within expected ranges
    assert 45.0 <= endpoints.lower <= 55.0
    assert 85.0 <= endpoints.upper <= 90.0

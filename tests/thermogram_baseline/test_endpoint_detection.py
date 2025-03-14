"""
Unit tests for the endpoint detection module.

These tests verify that the endpoint detection algorithm correctly
identifies optimal endpoints for thermogram baseline subtraction.
"""

import numpy as np
import polars as pl
import pytest

from thermogram_baseline.endpoint_detection import Endpoints, detect_endpoints


class TestEndpointDetection:
    """Tests for endpoint detection functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create synthetic thermogram data for testing
        # Use a wider temperature range and more points to accommodate
        # larger window sizes
        temp_range = np.linspace(45, 90, 900)  # Doubled number of points

        # Create a curve with multiple peaks
        peak1 = 0.3 * np.exp(-0.5 * ((temp_range - 63) / 2) ** 2)  # Peak ~ 63°C
        peak2 = 0.2 * np.exp(-0.5 * ((temp_range - 70) / 2) ** 2)  # Peak ~ 70°C
        peak3 = 0.15 * np.exp(-0.5 * ((temp_range - 77) / 2.5) ** 2)  # Peak ~ 77°C
        baseline = 0.02 * (temp_range - 65)  # Slight linear baseline
        dcp = peak1 + peak2 + peak3 + baseline + 0.02 * np.random.randn(len(temp_range))

        # Create a polars DataFrame
        self.data = pl.DataFrame({"Temperature": temp_range, "dCp": dcp})

        # Create a simpler case with VERY obvious endpoints - step function
        simple_temp = np.linspace(45, 90, 900)

        # Create a perfect step function with no noise at all
        simple_dcp = np.zeros_like(simple_temp)

        # Signal zone exactly from 60-80°C with much higher values
        mask_middle = (simple_temp >= 60) & (simple_temp <= 80)
        # A perfect step function: 0 outside, 1 inside the signal region
        simple_dcp[mask_middle] = 1.0
        simple_dcp[~mask_middle] = 0.0

        # No noise at all for this test
        self.simple_data = pl.DataFrame({"Temperature": simple_temp, "dCp": simple_dcp})

    def test_detect_endpoints_basic(self) -> None:
        """Test basic endpoint detection."""
        # Run endpoint detection with default parameters
        endpoints = detect_endpoints(self.data)

        # Check that the endpoints object has expected attributes
        assert isinstance(endpoints, Endpoints)
        assert hasattr(endpoints, "lower")
        assert hasattr(endpoints, "upper")
        assert hasattr(endpoints, "method")

        # Check that endpoints are within expected ranges
        assert 45 <= endpoints.lower <= 60
        assert 80 <= endpoints.upper <= 90
        assert endpoints.method == "innermost"

    def test_detect_endpoints_point_selection(self) -> None:
        """Test different point selection methods."""
        # Test innermost
        endpoints_inner = detect_endpoints(self.data, point_selection="innermost")
        assert endpoints_inner.method == "innermost"

        # Test outermost
        endpoints_outer = detect_endpoints(self.data, point_selection="outermost")
        assert endpoints_outer.method == "outermost"

        # For the test data:
        # - Innermost should be closer to the center (60-80°C)
        # - Outermost should be further from the center
        # Note: We're testing the relative positions, not absolute values
        assert endpoints_outer.lower <= endpoints_inner.lower
        assert endpoints_outer.upper >= endpoints_inner.upper

        # Test mid
        endpoints_mid = detect_endpoints(self.data, point_selection="mid")
        assert endpoints_mid.method == "mid"

    def test_detect_endpoints_window_size(self) -> None:
        """Test effect of window size on endpoint detection."""
        # Use more reasonable window sizes for the test data
        endpoints_small_w = detect_endpoints(self.data, w=30)
        endpoints_large_w = detect_endpoints(self.data, w=60)

        # We can't make absolute assertions about the relationship
        # since it depends on the data, but we can check that they're different
        # and still within valid ranges
        assert endpoints_small_w != endpoints_large_w

        # Check that the endpoints are still within valid ranges
        assert 45 <= endpoints_small_w.lower <= 60
        assert 80 <= endpoints_small_w.upper <= 90
        assert 45 <= endpoints_large_w.lower <= 60
        assert 80 <= endpoints_large_w.upper <= 90

    def test_detect_endpoints_exclusion_zone(self) -> None:
        """Test effect of exclusion zone on endpoint detection."""
        # Use narrower exclusion zones that will work with our test data
        endpoints_narrow = detect_endpoints(
            self.data, exclusion_lwr=65, exclusion_upr=75
        )
        endpoints_wide = detect_endpoints(self.data, exclusion_lwr=58, exclusion_upr=82)

        # Wider exclusion zone should result in endpoints further
        # from the central region
        assert endpoints_narrow.lower >= endpoints_wide.lower
        assert endpoints_narrow.upper <= endpoints_wide.upper

    def test_detect_endpoints_obvious_case(self) -> None:
        """Test endpoint detection with a simple step function case.

        Note: The endpoint detection algorithm is designed to find regions of
        minimal variance, not sharp transitions. For a step function, the algorithm
        will actually find stable regions *near* the transitions, not exactly at
        the transitions themselves.
        """
        # Create a clean step function for testing
        step_temp = np.linspace(45, 90, 900)
        step_dcp = np.zeros_like(step_temp)
        mask_middle = (step_temp >= 60) & (step_temp <= 80)
        step_dcp[mask_middle] = 1.0

        # Create a fresh DataFrame just for this test
        step_data = pl.DataFrame({"Temperature": step_temp, "dCp": step_dcp})

        # Run endpoint detection with smaller window and custom exclusion zone
        endpoints = detect_endpoints(
            step_data,
            w=15,
            exclusion_lwr=65,
            exclusion_upr=75,
            point_selection="innermost",  # Try to get closest to transition
        )

        print(
            f"Detected endpoints: lower={endpoints.lower:.2f}, \
            upper={endpoints.upper:.2f}"
        )

        # For step function data, the algorithm detects stable regions near transitions
        # The exact position depends on window size and other factors,
        # so we use a more realistic tolerance
        assert endpoints.lower < 60  # Should be below the transition point
        assert endpoints.upper > 75  # Should be above the exclusion zone

        # Check that endpoints are in reasonable ranges
        assert 45 < endpoints.lower < 60  # Between min temp and transition
        assert 75 < endpoints.upper < 90  # Between exclusion zone and max temp

    def test_detect_endpoints_invalid_inputs(self) -> None:
        """Test that endpoint detection raises appropriate errors for invalid inputs."""
        # Test invalid point selection
        with pytest.raises(ValueError):
            detect_endpoints(self.data, point_selection="invalid")  # type: ignore

        # Test exclusion zone outside data range
        with pytest.raises(ValueError):
            detect_endpoints(self.data, exclusion_lwr=30, exclusion_upr=40)

        with pytest.raises(ValueError):
            detect_endpoints(self.data, exclusion_lwr=95, exclusion_upr=100)

        # Test insufficient points for window size
        with pytest.raises(ValueError):
            detect_endpoints(self.data, w=1000)

        # Test missing required columns
        bad_data = pl.DataFrame(
            {"Temperature": [50, 60, 70], "WrongColumn": [0.1, 0.2, 0.3]}
        )
        with pytest.raises(ValueError):
            detect_endpoints(bad_data)

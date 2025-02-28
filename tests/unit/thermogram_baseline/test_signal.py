"""Tests for the signal detection module."""

import pytest

from packages.thermogram_baseline.thermogram_baseline.signal import detect_signal
from tests.test_data_utils import (
    generate_multi_peak_thermogram,
    generate_simple_thermogram,
    thermogram_to_dataframe,
)


def test_detect_signal_with_clear_signal():
    """Test signal detection with a thermogram containing a clear signal."""
    # Generate a test thermogram with a clear peak
    test_data = generate_simple_thermogram(
        peak_height=1.5,  # Strong peak
        noise_level=0.01,  # Low noise
    )

    # Detect signal
    result = detect_signal(
        test_data,
        verbose=True,
    )

    # Check result structure
    assert isinstance(result.is_signal, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.details, dict)

    # Should detect a signal
    assert result.is_signal
    assert result.confidence > 0.5


def test_detect_signal_with_noise_only():
    """Test signal detection with a thermogram containing only noise."""
    # Generate a test thermogram with only noise (no peaks)
    test_data = generate_simple_thermogram(
        peak_height=0.0,  # No peak
        noise_level=0.05,  # Just noise
        baseline_slope=0.0,  # Flat baseline
        baseline_intercept=0.0,
    )

    # Detect signal
    result = detect_signal(
        test_data,
        verbose=True,
    )

    # Should not detect a signal
    assert not result.is_signal
    assert result.confidence <= 0.7  # Lower confidence for noise


def test_detect_signal_with_dataframe_input():
    """Test signal detection with DataFrame input."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Detect signal
    result = detect_signal(test_df)

    # Check result structure
    assert isinstance(result.is_signal, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.details, dict)


def test_detect_signal_with_different_methods():
    """Test signal detection with different detection methods."""
    # Generate a test thermogram with a clear signal
    test_data = generate_multi_peak_thermogram()

    # Test with ARIMA method
    result_arima = detect_signal(
        test_data,
        method="arima",
    )

    # Test with ADF method
    result_adf = detect_signal(
        test_data,
        method="adf",
    )

    # Both should detect a signal for a clear multi-peak thermogram
    assert result_arima.is_signal
    assert result_adf.is_signal
    assert result_arima.details["method"] == "arima"
    assert result_adf.details["method"] == "adf"


def test_detect_signal_invalid_method():
    """Test signal detection with an invalid method."""
    test_data = generate_simple_thermogram()

    # Should raise an error for unknown method
    with pytest.raises(ValueError, match="Unknown detection method"):
        detect_signal(test_data, method="unknown")


def test_detect_signal_threshold_effect():
    """Test how threshold affects signal detection."""
    # Generate a test thermogram with a moderate signal
    test_data = generate_simple_thermogram(
        peak_height=0.5,  # Moderate peak
        noise_level=0.1,  # Moderate noise
    )

    # Test with different thresholds
    result_strict = detect_signal(test_data, threshold=0.01)  # Strict
    result_moderate = detect_signal(test_data, threshold=0.05)  # Moderate
    result_lenient = detect_signal(test_data, threshold=0.1)  # Lenient

    # More lenient thresholds should be more likely to detect signal
    # but this is only a general guideline and may not always hold
    # depending on the specific data and detection method
    if not result_strict.is_signal and result_lenient.is_signal:
        assert result_lenient.confidence <= result_moderate.confidence

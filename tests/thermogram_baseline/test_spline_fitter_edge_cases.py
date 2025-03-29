"""
Tests for SplineFitter focusing on edge cases.

These tests verify that the SplineFitter works correctly for challenging
data scenarios, such as few data points, duplicate x values, etc.
"""

import numpy as np
import pytest

from thermogram_baseline.spline_fitter import SplineFitter


@pytest.fixture
def fitter():
    """Create a SplineFitter instance for testing."""
    return SplineFitter(verbose=True)


def test_small_dataset(fitter):
    """Test with very few data points."""
    # Generate minimal dataset (just above the required minimum)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.5, 0.5, 1.0, 2.0])

    # Should not raise exceptions
    spline = fitter.fit_with_gcv(x, y, use_r=False)
    fitted = spline(x)

    # Basic sanity checks
    assert len(fitted) == len(x)
    assert np.all(np.isfinite(fitted))


def test_duplicate_x_values(fitter):
    """Test with duplicate x values."""
    # Create data with duplicate x values
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    y = np.array([1.0, 1.2, 1.5, 0.5, 1.0, 2.0, 1.8])

    # Should not raise exceptions
    spline = fitter.fit_with_gcv(x, y, use_r=False)

    # Test at original points
    fitted = spline(x)
    assert len(fitted) == len(x)

    # Test at new points
    x_new = np.linspace(1.0, 5.0, 10)
    fitted_new = spline(x_new)
    assert len(fitted_new) == len(x_new)
    assert np.all(np.isfinite(fitted_new))


def test_perfect_fit(fitter):
    """Test with data that can be perfectly fit by a spline."""
    # Generate data from a cubic polynomial (perfect fit for cubic spline)
    x = np.linspace(0, 10, 20)
    y = x**3 - 5 * x**2 + 3 * x - 1

    # Should fit very well with spar=0 (no smoothing)
    spline = fitter.fit_with_gcv(x, y, spar=0.0, use_r=False)
    fitted = spline(x)

    # Check for very close fit
    assert np.max(np.abs(fitted - y)) < 1e-10


def test_constant_data(fitter):
    """Test with constant y values."""
    # Generate constant data
    x = np.linspace(0, 10, 20)
    y = np.ones_like(x) * 5.0

    # Should not raise exceptions
    spline = fitter.fit_with_gcv(x, y, use_r=False)
    fitted = spline(x)

    # Should approximate a constant
    assert np.max(np.abs(fitted - 5.0)) < 0.1


def test_noisy_data(fitter):
    """Test with very noisy data."""
    # Generate noisy data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.5, size=len(x))

    # Should not raise exceptions
    spline = fitter.fit_with_gcv(x, y, use_r=False)
    fitted = spline(x)

    # Basic sanity checks
    assert len(fitted) == len(x)
    assert np.all(np.isfinite(fitted))

    # With high noise, spline should be smoother than data
    y_diff = np.diff(y)
    fitted_diff = np.diff(fitted)
    assert np.std(fitted_diff) < np.std(y_diff)


def test_insufficient_data(fitter):
    """Test with insufficient data points."""
    # Generate too few points for cubic spline
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 1.0])

    # Should raise ValueError
    with pytest.raises(ValueError):
        fitter.fit_with_gcv(x, y, use_r=False)


def test_extrapolation(fitter):
    """Test extrapolation outside the data range."""
    # Generate data
    x = np.linspace(1, 5, 20)
    y = x**2

    # Fit spline
    spline = fitter.fit_with_gcv(x, y, use_r=False)

    # Test extrapolation
    x_extra = np.array([0, 6])
    fitted_extra = spline(x_extra)

    # Should return finite values
    assert np.all(np.isfinite(fitted_extra))


def test_non_finite_values(fitter):
    """Test handling of non-finite values."""
    # Generate data with NaN and Inf
    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    y[3] = np.nan
    y[7] = np.inf

    # Should raise ValueError
    with pytest.raises(ValueError):
        fitter.fit_with_gcv(x, y, use_r=False)


def test_r_integration_fallback():
    """Test fallback from R to Python implementation."""
    # Create fitter that will try R first
    fitter = SplineFitter(verbose=True)

    # Generate simple data
    x = np.linspace(0, 10, 20)
    y = np.sin(x)

    # Test with use_r=True (should fallback if R not available)
    spline = fitter.fit_with_gcv(x, y, use_r=True)

    # Should not raise exceptions and return a valid spline
    fitted = spline(x)
    assert len(fitted) == len(x)
    assert np.all(np.isfinite(fitted))

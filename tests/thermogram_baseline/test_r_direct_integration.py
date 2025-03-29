"""
Tests for direct R integration through rpy2.

These tests verify that the direct R integration using rpy2 works correctly.
They are skipped if rpy2 is not available.
"""

import numpy as np
import pytest

from thermogram_baseline.spline_fitter import SplineFitter


@pytest.mark.r_validation
def test_r_spline_class():
    """Test that the RSpline wrapper class works correctly."""
    try:
        import rpy2.robjects.numpy2ri as numpy2ri

        # Activate conversion
        numpy2ri.activate()

        # Generate test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)

        # Create SplineFitter
        fitter = SplineFitter(verbose=True)

        # Fit using R
        r_spline = fitter.fit_with_gcv(x, y, use_r=True)

        # Check that we got the right type
        assert r_spline.__class__.__name__ == "RSpline"

        # Test prediction
        y_pred = r_spline(x)
        assert len(y_pred) == len(x)
        assert np.all(np.isfinite(y_pred))

        # Test prediction at new points
        x_new = np.linspace(-1, 11, 50)
        y_new = r_spline(x_new)
        assert len(y_new) == len(x_new)
        assert np.all(np.isfinite(y_new))

    except ImportError:
        pytest.skip("rpy2 not available")


@pytest.mark.r_validation
def test_r_integration_with_parameters():
    """Test R integration with different parameters."""
    try:
        import rpy2.robjects.numpy2ri as numpy2ri

        # Activate conversion
        numpy2ri.activate()

        # Generate test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)

        # Create SplineFitter
        fitter = SplineFitter(verbose=True)

        # Test with different spar values
        for spar in [0.0, 0.5, 1.0]:
            r_spline = fitter.fit_with_gcv(x, y, spar=spar, use_r=True)

            # Check spar value was passed correctly
            assert abs(r_spline.spar - spar) < 0.01

            # Predict
            y_pred = r_spline(x)
            assert len(y_pred) == len(x)
            assert np.all(np.isfinite(y_pred))

    except ImportError:
        pytest.skip("rpy2 not available")


@pytest.mark.r_validation
def test_r_fallback_behavior():
    """Test fallback behavior when R is not available."""
    # Create SplineFitter
    fitter = SplineFitter(verbose=True)

    # Override _r_available to simulate R not available
    original_r_available = fitter._r_available
    fitter._r_available = False

    try:
        # Generate test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)

        # Try to use R (should fall back to Python)
        spline = fitter.fit_with_gcv(x, y, use_r=True)

        # Should get UnivariateSpline not RSpline
        assert spline.__class__.__name__ != "RSpline"

        # Predict
        y_pred = spline(x)
        assert len(y_pred) == len(x)
        assert np.all(np.isfinite(y_pred))

    finally:
        # Restore original R availability
        fitter._r_available = original_r_available

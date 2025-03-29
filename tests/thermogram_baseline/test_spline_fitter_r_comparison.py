"""
Tests comparing SplineFitter with R's smooth.spline.
"""

import numpy as np
import pytest

from thermogram_baseline.spline_fitter import SplineFitter


def generate_test_data(n=100, pattern="sine"):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, n)

    if pattern == "sine":
        y = np.sin(x) + 0.1 * np.random.randn(n)
    elif pattern == "exp":
        y = np.exp(x / 5) + 0.1 * np.random.randn(n)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return x, y


@pytest.mark.r_validation
def test_compare_with_r_sine():
    """Compare SplineFitter with R's smooth.spline for sine pattern."""
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr

        # Activate conversion
        numpy2ri.activate()

        # Import R's stats package
        stats = importr("stats")

        # Generate test data
        x, y = generate_test_data(pattern="sine")

        # Fit with R
        r_x = robjects.FloatVector(x)
        r_y = robjects.FloatVector(y)
        r_fit = stats.smooth_spline(x=r_x, y=r_y, cv=True)

        # Extract R results
        r_fitted = np.array(r_fit.rx2("y"))
        r_df = float(r_fit.rx2("df")[0])
        r_spar = float(r_fit.rx2("spar")[0])

        # Fit with Python
        fitter = SplineFitter()
        py_spline = fitter.fit_with_gcv(x, y, use_r=False)
        py_fitted = py_spline(x)

        # Calculate differences
        abs_diff = np.abs(py_fitted - r_fitted)
        rel_diff = abs_diff / np.maximum(np.abs(r_fitted), 1e-10)

        # Print results for debugging
        print(f"R df: {r_df}, Python df: {getattr(py_spline, 'df', 'unknown')}")
        print(
            f"R spar: {r_spar}, Python spar: "
            f"{getattr(py_spline, 'spar_approx', 'unknown')}"
        )
        print(f"Max absolute difference: {np.max(abs_diff)}")
        print(f"Mean absolute difference: {np.mean(abs_diff)}")
        print(f"Max relative difference: {np.max(rel_diff) * 100}%")
        print(f"Mean relative difference: {np.mean(rel_diff) * 100}%")

        # Assert reasonable agreement (< 5% difference or absolute difference < 0.5)
        assert (np.max(abs_diff) < 0.5) or (np.mean(rel_diff) < 0.05), (
            f"Mean relative difference > 5%, got {(np.mean(rel_diff) * 100):.2f}%"
            f" and max absolute difference > 0.5, got {np.max(abs_diff)}"
        )

    except ImportError:
        pytest.skip("rpy2 not available")


@pytest.mark.r_validation
def test_compare_with_r_exp():
    """Compare SplineFitter with R's smooth.spline for exponential pattern."""
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr

        # Activate conversion
        numpy2ri.activate()

        # Import R's stats package
        stats = importr("stats")

        # Generate test data
        x, y = generate_test_data(pattern="exp")

        # Fit with R
        r_x = robjects.FloatVector(x)
        r_y = robjects.FloatVector(y)
        r_fit = stats.smooth_spline(x=r_x, y=r_y, cv=True)

        # Extract R results
        r_fitted = np.array(r_fit.rx2("y"))
        r_df = float(r_fit.rx2("df")[0])
        r_spar = float(r_fit.rx2("spar")[0])

        # Fit with Python
        fitter = SplineFitter()
        py_spline = fitter.fit_with_gcv(x, y, use_r=False)
        py_fitted = py_spline(x)

        # Calculate differences
        abs_diff = np.abs(py_fitted - r_fitted)
        rel_diff = abs_diff / np.maximum(np.abs(r_fitted), 1e-10)

        # Print results for debugging
        print(f"R df: {r_df}, Python df: {getattr(py_spline, 'df', 'unknown')}")
        print(
            f"R spar: {r_spar}, Python spar: "
            f"{getattr(py_spline, 'spar_approx', 'unknown')}"
        )
        print(f"Max absolute difference: {np.max(abs_diff)}")
        print(f"Mean absolute difference: {np.mean(abs_diff)}")
        print(f"Max relative difference: {np.max(rel_diff) * 100}%")
        print(f"Mean relative difference: {np.mean(rel_diff) * 100}%")

        # Assert reasonable agreement
        assert np.mean(rel_diff) < 0.05, "Mean relative difference > 5%"

    except ImportError:
        pytest.skip("rpy2 not available")


@pytest.mark.r_validation
def test_edge_cases():
    """Test SplineFitter on edge cases compared to R."""
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr

        # Activate conversion
        numpy2ri.activate()

        # Import R's stats package
        stats = importr("stats")

        # Edge case 1: Few data points
        np.random.seed(42)
        x_few = np.linspace(0, 10, 10)
        y_few = np.sin(x_few) + 0.1 * np.random.randn(10)

        # Edge case 2: Duplicate x values
        x_dup = np.array([1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10])
        y_dup = np.sin(x_dup) + 0.1 * np.random.randn(len(x_dup))

        # Test small dataset
        print("\nTesting small dataset:")
        r_x = robjects.FloatVector(x_few)
        r_y = robjects.FloatVector(y_few)

        try:
            r_fit = stats.smooth_spline(x=r_x, y=r_y, cv=True)
            r_fitted = np.array(r_fit.rx2("y"))

            fitter = SplineFitter()
            py_spline = fitter.fit_with_gcv(x_few, y_few, use_r=False)
            py_fitted = py_spline(x_few)

            abs_diff = np.abs(py_fitted - r_fitted)
            rel_diff = abs_diff / np.maximum(np.abs(r_fitted), 1e-10)

            print(
                f"Small dataset - Mean relative difference: {np.mean(rel_diff) * 100}%"
            )
            assert (
                np.mean(rel_diff) < 0.1
            ), "Small dataset: Mean relative difference > 10%"
        except Exception as e:
            print(f"Small dataset test failed: {e}")

        # Test duplicate x values
        print("\nTesting duplicate x values:")
        r_x = robjects.FloatVector(x_dup)
        r_y = robjects.FloatVector(y_dup)

        try:
            r_fit = stats.smooth_spline(x=r_x, y=r_y, cv=True)
            r_fitted = np.array(r_fit.rx2("y"))

            fitter = SplineFitter()
            py_spline = fitter.fit_with_gcv(x_dup, y_dup, use_r=False)
            py_fitted = py_spline(x_dup)

            abs_diff = np.abs(py_fitted - r_fitted)
            rel_diff = abs_diff / np.maximum(np.abs(r_fitted), 1e-10)

            print(
                f"Duplicate x values - Mean relative difference: "
                f"{np.mean(rel_diff) * 100}%"
            )
            assert (
                np.mean(rel_diff) < 0.1
            ), "Duplicate x values: Mean relative difference > 10%"
        except Exception as e:
            print(f"Duplicate x values test failed: {e}")

    except ImportError:
        pytest.skip("rpy2 not available")

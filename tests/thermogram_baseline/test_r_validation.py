"""
Tests that validate SplineFitter against R's smooth.spline output.

This module uses reference data generated by R's smooth.spline function
to validate that our Python SplineFitter produces equivalent results.
"""

from pathlib import Path
from typing import Dict, TypedDict

import numpy as np
import plotly.graph_objects as go  # type: ignore
import polars as pl
import pytest

from thermogram_baseline.spline_fitter import SplineFitter

# Path to R reference data directory
R_REFERENCE_DIR = Path("tests/data/r_reference")


class RParams(TypedDict):
    """Type definition for R parameters dictionary."""

    spar: float
    df: float
    lambda_: float  # Note: renamed from 'lambda' which is a Python keyword


class ReferenceData(TypedDict):
    """Type definition for reference data returned by load_reference_data."""

    x: np.ndarray
    y: np.ndarray
    r_fitted: np.ndarray
    r_params: RParams


def load_reference_data(pattern: str, size: str = "standard") -> ReferenceData:
    """
    Load R reference data for a specific pattern and size.

    Args:
        pattern: The pattern name (e.g., 'sine', 'peaks')
        size: The dataset size ('standard' or 'small')

    Returns:
        Dictionary containing input data, R-fitted values, and parameters

    Raises:
        pytest.skip: If reference data files are not found
    """
    input_file = R_REFERENCE_DIR / f"{size}_{pattern}_input.csv"
    fitted_file = R_REFERENCE_DIR / f"{size}_{pattern}_fitted.csv"
    params_file = R_REFERENCE_DIR / f"{size}_{pattern}_params.csv"

    if not input_file.exists() or not fitted_file.exists() or not params_file.exists():
        pytest.skip(f"Reference data files not found for {pattern} ({size})")

    input_data = pl.read_csv(input_file)
    fitted_data = pl.read_csv(fitted_file)
    params_data = pl.read_csv(params_file)

    return {
        "x": input_data["x"].to_numpy(),
        "y": input_data["y"].to_numpy(),
        "r_fitted": fitted_data["fitted"].to_numpy(),
        "r_params": {
            "spar": float(params_data["spar"][0]),
            "df": float(params_data["df"][0]),
            "lambda_": float(params_data["lambda"][0]),  # Note the underscore added
        },
    }


def compare_fits(
    x: np.ndarray,
    y: np.ndarray,
    r_fitted: np.ndarray,
    pattern: str,
    size: str = "standard",
    save_plot: bool = True,
) -> Dict[str, float]:
    """
    Compare Python SplineFitter with R smooth.spline fitted values.

    Args:
        x: Independent variable values
        y: Dependent variable values
        r_fitted: Fitted values from R's smooth.spline
        pattern: The pattern name for output filenames
        size: The dataset size ('standard' or 'small')
        save_plot: Whether to save comparison plots

    Returns:
        Dictionary with difference metrics between Python and R implementations
    """
    # Fit with Python implementation
    fitter = SplineFitter()
    spline = fitter.fit_with_gcv(x, y)
    py_fitted = spline(x)

    # Calculate differences
    abs_diff = np.abs(py_fitted - r_fitted)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Calculate relative differences (avoid division by zero)
    mask = np.abs(r_fitted) > 1e-10
    rel_diff = np.zeros_like(r_fitted)
    rel_diff[mask] = abs_diff[mask] / np.abs(r_fitted[mask])
    max_rel_diff = float(np.max(rel_diff)) * 100  # as percentage
    mean_rel_diff = float(np.mean(rel_diff)) * 100  # as percentage

    # Create comparison plot if requested
    if save_plot:
        # Create Plotly figure
        fig = go.Figure()

        # Add data points
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", name="Data", marker=dict(size=8, opacity=0.5)
            )
        )

        # Add R fitted values
        fig.add_trace(
            go.Scatter(
                x=x,
                y=r_fitted,
                mode="lines",
                name="R smooth.spline",
                line=dict(color="red", width=2),
            )
        )

        # Add Python fitted values
        fig.add_trace(
            go.Scatter(
                x=x,
                y=py_fitted,
                mode="lines",
                name="Python SplineFitter",
                line=dict(color="green", width=2, dash="dash"),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Comparison for {pattern} ({size})<br>Max Diff: \
                {max_abs_diff:.6f}, Mean Rel Diff: {mean_rel_diff:.4f}%",
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        # Create directory if it doesn't exist
        plots_dir = Path("tests/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save as HTML for interactive viewing
        fig.write_html(plots_dir / f"validation_{pattern}_{size}.html")

        # Save as PNG for static viewing
        fig.write_image(
            plots_dir / f"validation_{pattern}_{size}.png", width=1000, height=600
        )

    return {
        "max_absolute_difference": max_abs_diff,
        "mean_absolute_difference": mean_abs_diff,
        "max_relative_difference_percent": max_rel_diff,
        "mean_relative_difference_percent": mean_rel_diff,
    }


@pytest.mark.parametrize("pattern", ["sine", "exponential", "peaks", "flat", "noisy"])
def test_standard_fit(pattern: str) -> None:
    """
    Test SplineFitter against R reference data with standard size (n=100).

    Args:
        pattern: The pattern name to test
    """
    # Skip test if reference data doesn't exist (allows running tests without R data)
    if not (R_REFERENCE_DIR / f"standard_{pattern}_input.csv").exists():
        pytest.skip(f"Reference data not found for {pattern}")

    # Load reference data
    ref_data = load_reference_data(pattern, "standard")

    # Compare fits - extract arrays from the typed dictionary
    result = compare_fits(
        x=ref_data["x"],
        y=ref_data["y"],
        r_fitted=ref_data["r_fitted"],
        pattern=pattern,
        size="standard",
    )

    # Check acceptance criteria - typically <1% difference is acceptable
    # Looser criteria for very noisy data
    if pattern == "noisy":
        assert result["max_relative_difference_percent"] < 10.0
        assert result["mean_relative_difference_percent"] < 5.0
    else:
        assert result["max_relative_difference_percent"] < 2.0
        assert result["mean_relative_difference_percent"] < 1.0

    print(f"\nValidation results for {pattern}:")
    for key, value in result.items():
        print(f"  {key}: {value}")


@pytest.mark.parametrize("pattern", ["sine", "peaks"])
def test_small_dataset_fit(pattern: str) -> None:
    """
    Test SplineFitter against R reference data with small dataset (n=20).

    Args:
        pattern: The pattern name to test
    """
    # Skip test if reference data doesn't exist
    if not (R_REFERENCE_DIR / f"small_{pattern}_input.csv").exists():
        pytest.skip(f"Small reference data not found for {pattern}")

    # Load reference data
    ref_data = load_reference_data(pattern, "small")

    # Compare fits
    result = compare_fits(
        ref_data["x"], ref_data["y"], ref_data["r_fitted"], pattern, "small"
    )

    # Small datasets may have larger differences
    assert result["max_relative_difference_percent"] < 5.0
    assert result["mean_relative_difference_percent"] < 2.0


def test_thermogram_data() -> None:
    """
    Test SplineFitter against R reference data with thermogram-like data.
    """
    # Load thermogram reference data
    thermo_file = R_REFERENCE_DIR / "thermogram_fits.csv"
    params_file = R_REFERENCE_DIR / "thermogram_params.csv"

    if not thermo_file.exists() or not params_file.exists():
        pytest.skip("Thermogram reference data not found")

    thermo_data = pl.read_csv(thermo_file)
    # params_data = pl.read_csv(params_file) # Not used in this test

    # Extract data
    x = thermo_data["x"].to_numpy()
    y = thermo_data["y"].to_numpy()
    r_fitted_cv = thermo_data["fitted_cv_true"].to_numpy()

    # Compare fits
    result = compare_fits(x, y, r_fitted_cv, "thermogram", save_plot=True)

    # Check acceptance criteria
    assert result["max_relative_difference_percent"] < 2.0
    assert result["mean_relative_difference_percent"] < 1.0

    print("\nValidation results for thermogram data:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # This allows running the tests individually with visualization
    pattern = "peaks"  # Change to test different patterns

    ref_data = load_reference_data(pattern, "standard")
    result = compare_fits(
        ref_data["x"],
        ref_data["y"],
        ref_data["r_fitted"],
        pattern,
        "standard",
        save_plot=True,
    )

    print(f"\nValidation results for {pattern}:")
    for key, value in result.items():
        print(f"  {key}: {value}")

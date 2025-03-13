"""
Integration tests for the thermogram baseline workflow.

This module contains tests that verify the entire workflow from data loading
to endpoint detection to baseline subtraction works correctly as an integrated system.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore
import plotly.subplots as sp  # type: ignore
import polars as pl
import pytest

from thermogram_baseline.baseline import subtract_baseline
from thermogram_baseline.endpoint_detection import Endpoints, detect_endpoints
from thermogram_baseline.spline_fitter import SplineFitter


def create_synthetic_thermogram() -> pl.DataFrame:
    """
    Create a synthetic thermogram for testing.

    Creates a dataframe with Temperature and dCp columns that simulate
    a realistic thermogram with multiple peaks and a non-linear baseline.

    Returns:
        A Polars DataFrame with synthetic thermogram data
    """
    # Create temperature range
    temp_range = np.linspace(45, 90, 450)

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


# Add a helper function to safely get max value for annotations
def get_max_y_value(data_frame: pl.DataFrame, column_name: str) -> float:
    """Get the maximum value from a DataFrame column safely."""
    try:
        max_val = data_frame.select(pl.col(column_name).max()).item()
        # Ensure we're returning a float
        return float(max_val) if max_val is not None else 0.0
    except Exception:
        return 0.0


def test_complete_workflow(visualize: bool = False) -> None:
    """
    Test the complete workflow from data loading to baseline subtraction.

    This test performs the following steps:
    1. Create synthetic thermogram data
    2. Detect endpoints for baseline subtraction
    3. Subtract the baseline from the thermogram
    4. Verify the results meet expected criteria
    5. Optionally visualize the results

    Args:
        visualize: Whether to generate visualization files (default: False)
    """
    # Create test data
    data = create_synthetic_thermogram()

    # 1. Detect endpoints
    print("Step 1: Detecting endpoints...")
    endpoints: Endpoints = detect_endpoints(
        data, w=45, exclusion_lwr=60, exclusion_upr=80
    )

    # Check that endpoints were detected correctly
    assert (
        45 < endpoints.lower < 60
    ), f"Lower endpoint ({endpoints.lower}) should be between 45 and 60"
    assert (
        80 < endpoints.upper < 90
    ), f"Upper endpoint ({endpoints.upper}) should be between 80 and 90"

    print(
        f"Detected endpoints: lower={endpoints.lower:.2f}, upper={endpoints.upper:.2f}"
    )

    # 2. Subtract baseline
    print("Step 2: Subtracting baseline...")
    result = subtract_baseline(data, endpoints.lower, endpoints.upper)

    # Extract the DataFrame from the result, which could be either a DataFrame or tuple
    # with the DataFrame as the first element
    baseline_subtracted: pl.DataFrame
    if isinstance(result, tuple):
        baseline_subtracted = result[0]
    else:
        baseline_subtracted = result

    # Check that baseline subtraction produced valid data
    assert baseline_subtracted.height == data.height, "Row count should be preserved"
    assert (
        "Temperature" in baseline_subtracted.columns
    ), "Temperature column should be present"
    assert "dCp" in baseline_subtracted.columns, "dCp column should be present"

    # Calculate statistics on the subtracted data
    mean_dcp = baseline_subtracted.select(pl.mean("dCp")).item()
    abs_mean = abs(mean_dcp)

    # The mean should be close to zero after baseline subtraction
    assert (
        abs_mean < 0.1
    ), f"Mean dCp after baseline subtraction should be close to zero, got {mean_dcp}"

    print(f"Baseline subtraction complete. Mean dCp: {mean_dcp:.6f}")

    # 3. Visualize results using Plotly
    if visualize:
        # Create a 2x1 subplot figure
        fig = sp.make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "Original Thermogram with Detected Endpoints",
                "Baseline Subtracted Data",
            ],
            vertical_spacing=0.15,
        )

        # Original data - top subplot
        fig.add_trace(
            go.Scatter(
                x=data["Temperature"].to_numpy(),
                y=data["dCp"].to_numpy(),
                mode="markers",
                marker=dict(size=4, opacity=0.5),
                name="Original Data",
            ),
            row=1,
            col=1,
        )

        # Get max value for annotation positioning safely
        max_dcp_value = get_max_y_value(data, "dCp")

        # Add vertical lines for endpoints
        for endpoint, name in [
            (endpoints.lower, "Lower Endpoint"),
            (endpoints.upper, "Upper Endpoint"),
        ]:
            fig.add_vline(
                x=endpoint, line=dict(color="red", width=1, dash="dash"), row=1, col=1
            )

            # Add annotation for the endpoint
            fig.add_annotation(
                x=endpoint,
                y=max_dcp_value * 0.9,  # Use the safely calculated value
                text=name,
                showarrow=True,
                arrowhead=2,
                row=1,
                col=1,
            )

        # Baseline-subtracted data - bottom subplot
        fig.add_trace(
            go.Scatter(
                x=baseline_subtracted["Temperature"].to_numpy(),
                y=baseline_subtracted["dCp"].to_numpy(),
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="green"),
                name="Baseline Subtracted",
            ),
            row=2,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)

        # Update layout and formatting
        fig.update_layout(
            height=900,
            width=1000,
            title_text="Thermogram Baseline Subtraction Workflow",
            template="plotly_white",
            showlegend=True,
        )

        # Update x and y axis labels
        fig.update_xaxes(title_text="Temperature (°C)")
        fig.update_yaxes(title_text="dCp (kJ/mol·K)")

        # Create output directory if it doesn't exist
        out_dir = Path("tests/output")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save as HTML (interactive) and PNG (static)
        fig.write_html(out_dir / "integration_test_result.html")
        fig.write_image(out_dir / "integration_test_result.png")

        print(f"Visualizations saved to {out_dir}")

    print("Integration test completed successfully")


def test_splinefitter_with_workflow(visualize: bool = False) -> None:
    """
    Test the SplineFitter in the context of thermogram data.

    This test evaluates how well the spline fitter works with thermogram data:
    1. Create synthetic thermogram data
    2. Fit a spline to the data using SplineFitter
    3. Evaluate goodness of fit metrics
    4. Optionally visualize the results

    Args:
        visualize: Whether to generate visualization files (default: False)
    """
    # Create test data
    data = create_synthetic_thermogram()

    # Extract data for SplineFitter
    temps = data.select("Temperature").to_numpy().flatten()
    dcps = data.select("dCp").to_numpy().flatten()

    # Create SplineFitter and fit a spline
    fitter = SplineFitter()
    spline = fitter.fit_with_gcv(temps, dcps)

    # Get fitted values
    fitted_values = spline(temps)

    # Calculate goodness of fit metrics
    residuals = dcps - fitted_values
    rss = np.sum(residuals**2)
    tss = np.sum((dcps - np.mean(dcps)) ** 2)
    r_squared = 1 - (rss / tss)

    # The spline should fit the data reasonably well
    assert (
        r_squared > 0.7
    ), f"Spline should fit data well, but R² is only {r_squared:.4f}"

    print(f"SplineFitter test complete. R²: {r_squared:.4f}")

    # Visualize the spline fit if requested
    if visualize:
        fig = sp.make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Thermogram with Spline Fit", "Residuals"],
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
        )

        # Top plot: data and fitted line
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=dcps,
                mode="markers",
                marker=dict(size=4, opacity=0.5),
                name="Original Data",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=temps,
                y=fitted_values,
                mode="lines",
                line=dict(color="red", width=2),
                name="Fitted Spline",
            ),
            row=1,
            col=1,
        )

        # Bottom plot: residuals
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=residuals,
                mode="markers",
                marker=dict(color="blue", size=3, opacity=0.5),
                name="Residuals",
            ),
            row=2,
            col=1,
        )

        # Add zero line to residuals
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"), row=2, col=1)

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text=f"Spline Fitting Results (R² = {r_squared:.4f})",
            template="plotly_white",
            showlegend=True,
        )

        # Update axes
        fig.update_xaxes(title_text="Temperature (°C)")
        fig.update_yaxes(title_text="dCp (kJ/mol·K)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)

        # Create output directory if it doesn't exist
        out_dir = Path("tests/output")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save visualizations
        fig.write_html(out_dir / "spline_fit_result.html")
        fig.write_image(out_dir / "spline_fit_result.png")

        print(f"Spline fit visualizations saved to {out_dir}")


@pytest.mark.parametrize("visualize", [False])
def test_automated_workflow(visualize: bool) -> None:
    """
    Test the automated workflow function in the context of pytest.

    This ensures the test will run properly even when called from pytest,
    which doesn't capture print statements unless they fail.

    Args:
        visualize: Whether to generate visualization files
    """
    test_complete_workflow(visualize)


if __name__ == "__main__":
    # This allows running the tests directly for debugging with visualizations
    test_complete_workflow(visualize=True)
    test_splinefitter_with_workflow(visualize=True)

"""
Baseline subtraction module for thermogram analysis.

This module provides functionality for subtracting baselines from thermogram data,
replicating the functionality of the R ThermogramBaseline package.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go  # type: ignore
import plotly.subplots as sp  # type: ignore
import polars as pl
from scipy import stats  # type: ignore

from .spline_fitter import SplineFitter

# Create a singleton instance for performance
_spline_fitter = SplineFitter()


def subtract_baseline(
    data: pl.DataFrame,
    lwr_temp: float,
    upr_temp: float,
    plot: bool = False,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    use_r: Optional[bool] = None,
    verbose: bool = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, go.Figure]]:
    """
    Subtract baseline from thermogram data.

    This function fits splines to regions outside the signal zone, connects them
    through the signal region, and subtracts this baseline from the original data.

    Args:
        data: DataFrame with Temperature and dCp columns
        lwr_temp: Lower temperature endpoint for baseline subtraction
        upr_temp: Upper temperature endpoint for baseline subtraction
        plot: Whether to generate plots showing the baseline subtraction
        save_path: Path to save the plot HTML file (if None, will not save)
        show_plot: Whether to display the plot in the notebook/browser
        use_r: Whether to use R for spline fitting if available.
                If None, uses the THERMOGRAM_FORGE_USE_R environment variable.
        verbose: Whether to print verbose output

    Returns:
        If plot=False: DataFrame with Temperature and baseline-subtracted
                dCp columns
        If plot=True: Tuple of (DataFrame, Figure object)

    Raises:
        ValueError: If inputs don't meet requirements:
            - Data must contain 'Temperature' and 'dCp' columns
            - Endpoints must be within data range

    Examples:
        >>> import polars as pl
        >>> data = pl.read_csv("thermogram_data.csv")
        >>> endpoints = detect_endpoints(data)
        >>> # Without plotting
        >>> baseline_subtracted = subtract_baseline(
        ...     data,
        ...     lwr_temp=endpoints.lower,
        ...     upr_temp=endpoints.upper
        ... )
        >>> # With plotting
        >>> result, fig = subtract_baseline(
        ...     data,
        ...     lwr_temp=endpoints.lower,
        ...     upr_temp=endpoints.upper,
        ...     plot=True
        ... )
    """
    # Validate inputs
    if not all(col in data.columns for col in ["Temperature", "dCp"]):
        raise ValueError("Data must contain 'Temperature' and 'dCp' columns")

    # Check and adjust endpoints if too close to boundaries
    min_temp = data.select(pl.min("Temperature")).item()
    max_temp = data.select(pl.max("Temperature")).item()

    if lwr_temp <= min_temp:
        raise ValueError(
            f"Lower endpoint ({lwr_temp}) "
            f"must be greater than min temperature ({min_temp})"
        )

    if upr_temp >= max_temp:
        raise ValueError(
            f"Upper endpoint ({upr_temp}) "
            f"must be less than max temperature ({max_temp})"
        )

    # Use the global SplineFitter instance
    fitter = _spline_fitter

    # If verbose is True for this call, create a verbose fitter
    if verbose:
        fitter = SplineFitter(verbose=True)

    # Extract baseline regions
    lower_region = data.filter(pl.col("Temperature") < lwr_temp)
    upper_region = data.filter(pl.col("Temperature") > upr_temp)
    middle_region = data.filter(
        (pl.col("Temperature") >= lwr_temp) & (pl.col("Temperature") <= upr_temp)
    )

    # Convert to numpy for spline fitting
    lower_temp = lower_region.select("Temperature").to_numpy().flatten()
    lower_dcp = lower_region.select("dCp").to_numpy().flatten()
    upper_temp = upper_region.select("Temperature").to_numpy().flatten()
    upper_dcp = upper_region.select("dCp").to_numpy().flatten()

    # Fit splines to lower and upper regions using SplineFitter
    if len(lower_temp) > 3:  # Need at least 4 points for cubic spline
        spline_lower = fitter.fit_with_gcv(lower_temp, lower_dcp, use_r=use_r)
        lower_fitted = spline_lower(lower_temp)
        lower_region = lower_region.with_columns(pl.Series("baseline", lower_fitted))
    else:
        # Not enough points, use linear fit
        if len(lower_temp) >= 2:
            slope, intercept, _, _, _ = stats.linregress(lower_temp, lower_dcp)
            lower_fitted = slope * lower_temp + intercept
            lower_region = lower_region.with_columns(
                pl.Series("baseline", lower_fitted)
            )
        else:
            # Can't fit with just 1 point, use the value
            lower_fitted = lower_dcp
            lower_region = lower_region.with_columns(
                pl.Series("baseline", lower_fitted)
            )

    if len(upper_temp) > 3:  # Need at least 4 points for cubic spline
        spline_upper = fitter.fit_with_gcv(upper_temp, upper_dcp, use_r=use_r)
        upper_fitted = spline_upper(upper_temp)
        upper_region = upper_region.with_columns(pl.Series("baseline", upper_fitted))
    else:
        # Not enough points, use linear fit
        if len(upper_temp) >= 2:
            slope, intercept, _, _, _ = stats.linregress(upper_temp, upper_dcp)
            upper_fitted = slope * upper_temp + intercept
            upper_region = upper_region.with_columns(
                pl.Series("baseline", upper_fitted)
            )
        else:
            # Can't fit with just 1 point, use the value
            upper_fitted = upper_dcp
            upper_region = upper_region.with_columns(
                pl.Series("baseline", upper_fitted)
            )

    # Find connection points
    if len(lower_region) > 0 and len(upper_region) > 0:
        lower_connection = lower_region.filter(
            pl.col("Temperature") == lower_region.select(pl.max("Temperature")).item()
        )
        upper_connection = upper_region.filter(
            pl.col("Temperature") == upper_region.select(pl.min("Temperature")).item()
        )

        # Extract connection points
        connect_temps = [
            lower_connection.select("Temperature").item(),
            upper_connection.select("Temperature").item(),
        ]
        connect_baselines = [
            lower_connection.select("baseline").item(),
            upper_connection.select("baseline").item(),
        ]

        # Linear interpolation through middle region
        middle_temp = middle_region.select("Temperature").to_numpy().flatten()

        # Handle case where endpoints are the same
        if abs(connect_temps[1] - connect_temps[0]) < 1e-10:
            middle_baseline = np.full_like(middle_temp, connect_baselines[0])
        else:
            # Linear interpolation
            slope = (connect_baselines[1] - connect_baselines[0]) / (
                connect_temps[1] - connect_temps[0]
            )
            intercept = connect_baselines[0] - slope * connect_temps[0]
            middle_baseline = slope * middle_temp + intercept

        middle_region = middle_region.with_columns(
            pl.Series("baseline", middle_baseline)
        )
    else:
        # Handle edge case if one region is empty
        if len(lower_region) == 0:
            # Just extend upper region baseline
            middle_temp = middle_region.select("Temperature").to_numpy().flatten()
            if len(upper_temp) >= 2:
                # Use regression if possible
                slope, intercept, _, _, _ = stats.linregress(upper_temp, upper_dcp)
                middle_baseline = slope * middle_temp + intercept
            else:
                # Use constant otherwise
                middle_baseline = np.full_like(middle_temp, upper_dcp.mean())
            middle_region = middle_region.with_columns(
                pl.Series("baseline", middle_baseline)
            )
        elif len(upper_region) == 0:
            # Just extend lower region baseline
            middle_temp = middle_region.select("Temperature").to_numpy().flatten()
            if len(lower_temp) >= 2:
                # Use regression if possible
                slope, intercept, _, _, _ = stats.linregress(lower_temp, lower_dcp)
                middle_baseline = slope * middle_temp + intercept
            else:
                # Use constant otherwise
                middle_baseline = np.full_like(middle_temp, lower_dcp.mean())
            middle_region = middle_region.with_columns(
                pl.Series("baseline", middle_baseline)
            )

    # Combine all regions
    combined = pl.concat([lower_region, middle_region, upper_region])

    # Subtract baseline from dCp
    result = combined.with_columns(
        (pl.col("dCp") - pl.col("baseline")).alias("dCp_subtracted")
    )

    # Keep only Temperature and dCp_subtracted columns
    result = result.select(["Temperature", "dCp_subtracted"]).rename(
        {"dCp_subtracted": "dCp"}
    )

    # Ensure result is sorted by Temperature
    result = result.sort("Temperature")

    # Generate plots if requested
    if plot:
        fig = plot_baseline_subtraction(
            data, lower_region, middle_region, upper_region, result
        )

        # Save the figure if a path is provided
        if save_path:
            # Make sure parent directories exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            print(f"Plot saved to {save_path}")

        # Show the figure if requested
        if show_plot:
            fig.show()

        return result, fig

    return result


def plot_baseline_subtraction(
    original: pl.DataFrame,
    lower: pl.DataFrame,
    middle: pl.DataFrame,
    upper: pl.DataFrame,
    result: pl.DataFrame,
) -> go.Figure:
    """
    Plot baseline subtraction results using Plotly.

    Args:
        original: Original data
        lower: Lower region with fitted baseline
        middle: Middle region with fitted baseline
        upper: Upper region with fitted baseline
        result: Baseline-subtracted result

    Returns:
        Plotly figure object with the baseline subtraction visualization
    """
    # Create a 2x1 subplot
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Raw Curve with Baseline Overlay", "Baseline Subtracted Data"],
        vertical_spacing=0.2,
        row_heights=[0.6, 0.4],
    )

    # Add the original data to the first subplot
    fig.add_trace(
        go.Scatter(
            x=original["Temperature"].to_numpy(),
            y=original["dCp"].to_numpy(),
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name="Raw Data",
        ),
        row=1,
        col=1,
    )

    # Add baseline segments
    # Lower region baseline
    if "baseline" in lower.columns:
        fig.add_trace(
            go.Scatter(
                x=lower["Temperature"].to_numpy(),
                y=lower["baseline"].to_numpy(),
                mode="lines",
                line=dict(color="red", width=2),
                name="Lower Baseline",
            ),
            row=1,
            col=1,
        )

    # Middle region baseline
    if "baseline" in middle.columns:
        show_legend = "baseline" not in lower.columns
        fig.add_trace(
            go.Scatter(
                x=middle["Temperature"].to_numpy(),
                y=middle["baseline"].to_numpy(),
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Middle Baseline",
                showlegend=show_legend,
            ),
            row=1,
            col=1,
        )

    # Upper region baseline
    if "baseline" in upper.columns:
        show_legend = (
            "baseline" not in lower.columns and "baseline" not in middle.columns
        )
        fig.add_trace(
            go.Scatter(
                x=upper["Temperature"].to_numpy(),
                y=upper["baseline"].to_numpy(),
                mode="lines",
                line=dict(color="red", width=2),
                name="Upper Baseline",
                showlegend=show_legend,
            ),
            row=1,
            col=1,
        )

    # Add vertical lines at transition points where the regions connect
    if "baseline" in lower.columns and "baseline" in middle.columns:
        transition_temp = (
            lower.filter(
                pl.col("Temperature") == lower.select(pl.max("Temperature")).item()
            )
            .select("Temperature")
            .item()
        )

        fig.add_vline(
            x=transition_temp,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=1,
        )

        fig.add_annotation(
            x=transition_temp,
            y=original.select(pl.max("dCp")).item() * 0.9,
            text="Lower Endpoint",
            showarrow=True,
            arrowhead=2,
            row=1,
            col=1,
        )

    if "baseline" in upper.columns and "baseline" in middle.columns:
        transition_temp = (
            upper.filter(
                pl.col("Temperature") == upper.select(pl.min("Temperature")).item()
            )
            .select("Temperature")
            .item()
        )

        fig.add_vline(
            x=transition_temp,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=1,
        )

        fig.add_annotation(
            x=transition_temp,
            y=original.select(pl.max("dCp")).item() * 0.8,
            text="Upper Endpoint",
            showarrow=True,
            arrowhead=2,
            row=1,
            col=1,
        )

    # Add the baseline-subtracted data to the second subplot
    fig.add_trace(
        go.Scatter(
            x=result["Temperature"].to_numpy(),
            y=result["dCp"].to_numpy(),
            mode="markers",
            marker=dict(size=4, opacity=0.5, color="green"),
            name="Baseline Subtracted",
        ),
        row=2,
        col=1,
    )

    # Add a horizontal line at y=0 for reference
    fig.add_shape(
        type="line",
        x0=result.select(pl.min("Temperature")).item(),
        y0=0,
        x1=result.select(pl.max("Temperature")).item(),
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="solid",
        ),
        opacity=0.3,
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Thermogram Baseline Subtraction",
        template="plotly_white",
        hovermode="closest",
        # Move legend to middle right of plot
        legend=dict(yanchor="top", y=0.5, xanchor="right", x=0.95),
    )

    # Update axes
    fig.update_xaxes(
        title_text="Temperature (°C)", showgrid=True, gridwidth=1, gridcolor="lightgray"
    )

    fig.update_yaxes(
        title_text="dCp (kJ/mol·K)", showgrid=True, gridwidth=1, gridcolor="lightgray"
    )

    return fig

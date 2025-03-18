"""
Interpolation module for thermogram data analysis.

This module implements interpolation of thermogram data onto a uniform
temperature grid using spline fitting.
"""

from typing import Optional

import numpy as np
import polars as pl

from thermogram_baseline.spline_fitter import SplineFitter


def interpolate_thermogram(
    data: pl.DataFrame, grid_temp: Optional[np.ndarray] = None, plot: bool = False
) -> pl.DataFrame:
    """
    Interpolate thermogram data onto a fixed temperature grid.

    This function fits a spline to the thermogram data and interpolates it onto
    a uniform temperature grid, using the same spline fitting approach as the
    baseline subtraction to maintain consistency with the R implementation.

    Args:
        data: DataFrame with Temperature and dCp columns
        grid_temp: Array of temperatures for interpolation. If None, defaults to
            numpy.arange(45, 90.1, 0.1)
        plot: Whether to generate and display a plot of the interpolated data

    Returns:
        DataFrame with Temperature and interpolated dCp columns
    """
    # Validate inputs
    if not all(col in data.columns for col in ["Temperature", "dCp"]):
        raise ValueError("Data must contain 'Temperature' and 'dCp' columns")

    # Create default grid if not provided
    if grid_temp is None:
        grid_temp = np.arange(45, 90.1, 0.1)

    # Extract data
    temp = data.select("Temperature").to_numpy().flatten()
    dcp = data.select("dCp").to_numpy().flatten()

    # Use the SplineFitter to create a spline (maintains consistency with R)
    fitter = SplineFitter()
    spline = fitter.fit_with_gcv(temp, dcp)

    # Predict values at the grid points
    interpolated_dcp = spline(grid_temp)

    # Create result DataFrame
    result = pl.DataFrame({"Temperature": grid_temp, "dCp": interpolated_dcp})

    # Generate plot if requested
    if plot:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Original data
            fig.add_trace(
                go.Scatter(
                    x=temp,
                    y=dcp,
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                    name="Original Data",
                )
            )

            # Interpolated data
            fig.add_trace(
                go.Scatter(
                    x=grid_temp,
                    y=interpolated_dcp,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Interpolated",
                )
            )

            fig.update_layout(
                title="Thermogram Interpolation",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp (kJ/mol·K)",
                template="plotly_white",
            )

            # If we have export utility, use it to save figure safely
            try:
                from thermogram_baseline.util.plotly_export import export_plotly_image

                export_plotly_image(fig, "interpolation_plot.png")
            except ImportError:
                pass

            fig.show()
        except ImportError:
            import warnings

            warnings.warn(
                "Plotly is required for plotting. Install with 'pip install plotly'"
            )

    return result


class ThermogramInterpolator:
    """Class for interpolating thermogram data."""

    def __init__(self) -> None:
        """Initialize ThermogramInterpolator."""
        self.fitter = SplineFitter()

    def interpolate(
        self,
        data: pl.DataFrame,
        grid_temp: Optional[np.ndarray] = None,
        plot: bool = False,
    ) -> pl.DataFrame:
        """
        Interpolate thermogram data onto a fixed temperature grid.

        Args:
            data: DataFrame with Temperature and dCp columns
            grid_temp: Array of temperatures for interpolation
            plot: Whether to generate and display plots

        Returns:
            DataFrame with interpolated data
        """
        return interpolate_thermogram(data, grid_temp, plot)

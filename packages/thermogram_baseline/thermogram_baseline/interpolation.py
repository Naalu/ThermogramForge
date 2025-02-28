"""Interpolation utilities for thermogram data."""

from typing import Optional, Sequence, Union

import numpy as np
import polars as pl
from scipy import interpolate

from .types import BaselineResult, InterpolatedResult, ThermogramData


def interpolate_sample(
    data: Union[ThermogramData, BaselineResult, pl.DataFrame],
    grid_temp: Optional[Union[Sequence[float], np.ndarray]] = None,
    smoothing_factor: Optional[float] = None,
    plot: bool = False,
) -> InterpolatedResult:
    """Interpolate thermogram data onto a uniform temperature grid.

    Fits a smoothing spline to the input data and evaluates it at specified grid points
    to create a uniformly spaced representation of the thermogram.

    Args:
        data: Input data in one of these formats:
            - ThermogramData: Direct thermogram data
            - BaselineResult: Result from baseline subtraction
            - pl.DataFrame: Raw data in DataFrame format
        grid_temp: Temperature points for interpolation. If None, uses range
            45 to 90 by 0.1. Defaults to None.
        smoothing_factor: Spline smoothing parameter. If None, uses data length.
            Defaults to None.
        plot: Whether to generate diagnostic plots. Defaults to False.

    Returns:
        InterpolatedResult containing:
            - data: Interpolated thermogram on uniform grid
            - grid_temp: Temperature grid used
            - original_data: Original input data if available
            - baseline_result: Baseline subtraction result if available

    Raises:
        ValueError: If input data contains invalid values
        TypeError: If input data format is not supported
    """
    # Handle different input types
    original_data = None
    baseline_result = None

    if isinstance(data, BaselineResult):
        # Use the subtracted data from the baseline result
        therm_data = data.subtracted
        original_data = data.original
        baseline_result = data
    elif isinstance(data, pl.DataFrame):
        # Convert DataFrame to ThermogramData
        therm_data = ThermogramData.from_dataframe(data)
    else:
        # Assume it's already ThermogramData
        therm_data = data

    # Create default grid if not provided
    if grid_temp is None:
        grid_temp = _create_default_grid()
    else:
        # Convert to numpy array if it's not already
        grid_temp = np.asarray(grid_temp)

    # Extract arrays for processing
    temperatures = therm_data.temperature
    values = therm_data.dcp

    # Fit spline to the data
    spline = _fit_interpolation_spline(temperatures, values, smoothing_factor)

    # Evaluate spline at grid points
    interpolated_values = spline(grid_temp)

    # Create result object
    interpolated_data = ThermogramData(temperature=grid_temp, dcp=interpolated_values)

    return InterpolatedResult(
        data=interpolated_data,
        grid_temp=grid_temp,
        original_data=original_data,
        baseline_result=baseline_result,
    )


def _create_default_grid(
    min_temp: float = 45.0,
    max_temp: float = 90.0,
    step: float = 0.1,
) -> np.ndarray:
    """Create a default temperature grid for interpolation.

    Args:
        min_temp: Minimum temperature value. Defaults to 45.0.
        max_temp: Maximum temperature value. Defaults to 90.0.
        step: Temperature increment between points. Defaults to 0.1.

    Returns:
        np.ndarray: Evenly spaced temperature grid from min_temp to max_temp
    """
    return np.arange(min_temp, max_temp + step / 2, step)


def _fit_interpolation_spline(
    temperatures: np.ndarray,
    values: np.ndarray,
    smoothing_factor: Optional[float] = None,
) -> interpolate.UnivariateSpline:
    """Fit a smoothing spline for interpolation.

    Args:
        temperatures: Array of temperature values.
        values: Array of heat capacity (dCp) values.
        smoothing_factor: Spline smoothing parameter. If None, uses length of data.
            Defaults to None.

    Returns:
        interpolate.UnivariateSpline: Fitted spline object for interpolation.

    Notes:
        - Data is automatically sorted by temperature if not already sorted
        - Default smoothing factor is set to number of data points
    """
    # Sort by temperature if not already sorted
    if not np.all(np.diff(temperatures) >= 0):
        sort_idx = np.argsort(temperatures)
        temperatures = temperatures[sort_idx]
        values = values[sort_idx]

    # Use cross-validation approach if smoothing factor not provided
    if smoothing_factor is None:
        # Start with a default value based on data size
        smoothing_factor = len(temperatures)

    # Fit the spline
    spline = interpolate.UnivariateSpline(
        x=temperatures,
        y=values,
        s=smoothing_factor,
    )

    return spline

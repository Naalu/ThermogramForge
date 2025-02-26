"""Baseline subtraction algorithms for thermogram data."""

from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import interpolate

from .types import BaselineResult, Endpoints, ThermogramData


def subtract_baseline(
    data: Union[ThermogramData, pl.DataFrame],
    lower_temp: float,
    upper_temp: float,
    smoothing_factor: Optional[float] = None,
    plot: bool = False,
) -> BaselineResult:
    """
    Subtract baseline from thermogram data using the specified endpoints.

    This function first fits splines to the regions outside the endpoints,
    then connects these regions with a straight line to create a complete
    baseline, which is then subtracted from the original data.

    Args:
        data: Thermogram data to process
        lower_temp: Lower temperature endpoint for baseline
        upper_temp: Upper temperature endpoint for baseline
        smoothing_factor: Optional smoothing factor for spline fitting
        plot: Whether to generate plots (will be implemented separately)

    Returns:
        BaselineResult containing original data, baseline, and subtracted result
    """
    # Convert input to ThermogramData if it's a DataFrame
    if isinstance(data, pl.DataFrame):
        therm_data = ThermogramData.from_dataframe(data)
    else:
        therm_data = data

    # Extract arrays for processing
    temperatures = therm_data.temperature
    values = therm_data.dcp

    # Validate endpoints
    min_temp = min(temperatures)
    max_temp = max(temperatures)

    if lower_temp < min_temp + 1:
        lower_temp = min_temp + 1

    if upper_temp > max_temp - 1:
        upper_temp = max_temp - 1

    # Extract regions for processing
    lower_mask = temperatures < lower_temp
    upper_mask = temperatures > upper_temp
    mid_mask = ~(lower_mask | upper_mask)

    lower_region_temps = temperatures[lower_mask]
    lower_region_values = values[lower_mask]

    upper_region_temps = temperatures[upper_mask]
    upper_region_values = values[upper_mask]

    mid_region_temps = temperatures[mid_mask]

    # Fit splines to lower and upper regions
    lower_spline = _fit_spline_to_region(
        lower_region_temps, lower_region_values, smoothing_factor
    )
    upper_spline = _fit_spline_to_region(
        upper_region_temps, upper_region_values, smoothing_factor
    )

    # Get spline values at endpoints
    lower_endpoint_value = lower_spline(lower_temp)
    upper_endpoint_value = upper_spline(upper_temp)

    # Connect endpoints with a straight line
    baseline_mid_values = _create_connecting_line(
        lower_temp,
        lower_endpoint_value,
        upper_temp,
        upper_endpoint_value,
        mid_region_temps,
    )

    # Create complete baseline
    baseline_lower_values = lower_spline(lower_region_temps)
    baseline_upper_values = upper_spline(upper_region_temps)

    # Combine baseline segments
    baseline_temps = np.concatenate(
        [lower_region_temps, mid_region_temps, upper_region_temps]
    )
    baseline_values = np.concatenate(
        [baseline_lower_values, baseline_mid_values, baseline_upper_values]
    )

    # Sort baseline by temperature (important if regions weren't already sorted)
    sort_idx = np.argsort(baseline_temps)
    baseline_temps = baseline_temps[sort_idx]
    baseline_values = baseline_values[sort_idx]

    # Create baseline object
    baseline = ThermogramData(temperature=baseline_temps, dcp=baseline_values)

    # Subtract baseline from original data
    subtracted_values = np.zeros_like(values)
    for i, temp in enumerate(temperatures):
        # Find closest temperature in baseline
        closest_idx = np.abs(baseline_temps - temp).argmin()
        subtracted_values[i] = values[i] - baseline_values[closest_idx]

    subtracted = ThermogramData(temperature=temperatures, dcp=subtracted_values)

    # Create endpoints object
    endpoints = Endpoints(lower=lower_temp, upper=upper_temp)

    # Return result
    return BaselineResult(
        original=therm_data,
        baseline=baseline,
        subtracted=subtracted,
        endpoints=endpoints,
    )


def _fit_spline_to_region(
    temperatures: np.ndarray,
    values: np.ndarray,
    smoothing_factor: Optional[float] = None,
) -> interpolate.UnivariateSpline:
    """
    Fit a smoothing spline to a region of the thermogram.

    Args:
        temperatures: Temperature values
        values: dCp values
        smoothing_factor: Optional smoothing factor

    Returns:
        Fitted spline object
    """
    # Sort by temperature if not already sorted
    if not np.all(np.diff(temperatures) >= 0):
        sort_idx = np.argsort(temperatures)
        temperatures = temperatures[sort_idx]
        values = values[sort_idx]

    # Determine smoothing factor if not provided
    if smoothing_factor is None:
        # Default to cross-validation approach
        smoothing_factor = len(temperatures)

    # Fit the spline
    spline = interpolate.UnivariateSpline(
        x=temperatures,
        y=values,
        s=smoothing_factor,
    )

    return spline


def _create_connecting_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x_points: np.ndarray,
) -> np.ndarray:
    """
    Create a straight line connecting two points evaluated at given x positions.

    Args:
        x1: x-coordinate of first point
        y1: y-coordinate of first point
        x2: x-coordinate of second point
        y2: y-coordinate of second point
        x_points: x-coordinates to evaluate the line at

    Returns:
        y-values of the line at the requested x-coordinates
    """
    # Linear function: y = m*x + b
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Evaluate at requested points
    y_points = slope * x_points + intercept

    return y_points

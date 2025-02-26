"""Main thermogram baseline processing functions."""

from typing import Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from .detection import detect_endpoints
from .interpolation import interpolate_sample
from .subtraction import subtract_baseline
from .types import InterpolatedResult, ThermogramData


def auto_baseline(
    data: Union[ThermogramData, pl.DataFrame],
    window_size: int = 90,
    exclusion_lower: float = 60,
    exclusion_upper: float = 80,
    grid_temp: Optional[Union[Sequence[float], np.ndarray]] = None,
    point_selection: Literal["innermost", "outmost", "mid"] = "innermost",
    plot: bool = False,
    verbose: bool = False,
) -> InterpolatedResult:
    """
    Perform automated baseline detection, subtraction and interpolation.

    This is the main function that combines endpoint detection, baseline subtraction,
    and interpolation to a uniform grid. It implements the complete workflow for
    thermogram baseline processing.

    Args:
        data: Thermogram data to process
        window_size: Window size for endpoint detection
        exclusion_lower: Lower bound of the exclusion window
        exclusion_upper: Upper bound of the exclusion window
        grid_temp: Temperature grid for interpolation (default: 45 to 90 by 0.1)
        point_selection: Method for endpoint selection
        plot: Whether to generate plots
        verbose: Whether to print progress information

    Returns:
        InterpolatedResult containing the processed data
    """
    if verbose:
        print("Step 1: Detecting endpoints...")

    # Detect endpoints for baseline subtraction
    endpoints = detect_endpoints(
        data=data,
        window_size=window_size,
        exclusion_lower=exclusion_lower,
        exclusion_upper=exclusion_upper,
        point_selection=point_selection,
        verbose=verbose,
    )

    if verbose:
        print(f"Endpoints detected: lower={endpoints.lower}, upper={endpoints.upper}")
        print("Step 2: Subtracting baseline...")

    # Subtract baseline using detected endpoints
    baseline_result = subtract_baseline(
        data=data, lower_temp=endpoints.lower, upper_temp=endpoints.upper, plot=plot
    )

    if verbose:
        print("Baseline subtracted successfully")
        print("Step 3: Interpolating to uniform grid...")

    # Interpolate to uniform grid
    interpolated_result = interpolate_sample(
        data=baseline_result, grid_temp=grid_temp, plot=plot
    )

    if verbose:
        print("Interpolation complete")
        print("Baseline processing workflow completed successfully")

    return interpolated_result

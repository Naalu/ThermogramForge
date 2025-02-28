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
    """Performs automated baseline detection, subtraction and interpolation.

    This function implements the complete workflow for processing thermogram data:
    1. Detects baseline endpoints
    2. Subtracts the baseline
    3. Interpolates to a uniform temperature grid

    Args:
        data (Union[ThermogramData, pl.DataFrame]): Thermogram data to process, either
            as a ThermogramData object or polars DataFrame.
        window_size (int, optional): Size of sliding window for endpoint detection.
            Defaults to 90.
        exclusion_lower (float, optional): Lower temperature bound of exclusion window.
            Defaults to 60.
        exclusion_upper (float, optional): Upper temperature bound of exclusion window.
            Defaults to 80.
        grid_temp (Optional[Union[Sequence[float], np.ndarray]], optional): Temperature
            grid points for interpolation. If None, uses range 45 to 90 by 0.1.
            Defaults to None.
        point_selection (Literal["innermost", "outmost", "mid"], optional): Method for
            selecting endpoints:
            - innermost: Points closest to exclusion window
            - outmost: Points farthest from exclusion window
            - mid: Points in middle of candidate ranges
            Defaults to "innermost".
        plot (bool, optional): Whether to generate diagnostic plots. Defaults to False.
        verbose (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        InterpolatedResult: Processed data containing:
            - temp: Temperature values on uniform grid
            - cp: Processed heat capacity values
            - baseline: Calculated baseline
            - endpoints: Detected baseline endpoints

    Raises:
        ValueError: If exclusion_upper <= exclusion_lower or no valid endpoints found
        TypeError: If input data is not ThermogramData or polars DataFrame

    Examples:
        >>> from thermogram_baseline import auto_baseline
        >>> result = auto_baseline(data, window_size=100, plot=True)
        >>> processed_cp = result.cp
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
        data=data,
        lower_temp=endpoints.lower,
        upper_temp=endpoints.upper,
        method=point_selection,
        plot=plot,
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

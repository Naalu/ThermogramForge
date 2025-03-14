"""
Endpoint detection module for thermogram data analysis.

This module implements algorithms to automatically detect optimal endpoints
for thermogram baseline subtraction by scanning from temperature extremes
toward an exclusion zone to find regions with minimum variance.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from thermogram_baseline.spline_fitter import SplineFitter


@dataclass
class Endpoints:
    """Container for baseline endpoint information.

    Attributes:
        lower: Lower temperature endpoint (°C)
        upper: Upper temperature endpoint (°C)
        method: Method used for endpoint selection
    """

    lower: float
    upper: float
    method: Literal["innermost", "outermost", "mid"]


def detect_endpoints(
    data: pl.DataFrame,
    w: int = 90,
    exclusion_lwr: float = 60,
    exclusion_upr: float = 80,
    point_selection: Literal["innermost", "outermost", "mid"] = "innermost",
    explicit: bool = False,
) -> Endpoints:
    """
    Detect optimal endpoints for thermogram baseline subtraction.

    Args:
        data: DataFrame with Temperature and dCp columns
        w: Window size for variance calculation
        exclusion_lwr: Lower bound of the exclusion window
        exclusion_upr: Upper bound of the exclusion window
        point_selection: Method for endpoint selection ("innermost", "outermost", "mid")
        explicit: Whether to print progress messages

    Returns:
        Endpoints object containing lower and upper temperature endpoints
        and the method used for selection

    Raises:
        ValueError: If inputs don't meet requirements:
            - Data must contain 'Temperature' and 'dCp' columns
            - Exclusion zone must be within data range
            - Not enough points outside exclusion zone
            - Invalid point_selection value
    """
    # Validate inputs
    if not all(col in data.columns for col in ["Temperature", "dCp"]):
        raise ValueError("Data must contain 'Temperature' and 'dCp' columns")

    # Check if point selection is valid
    valid_point_selections = ["innermost", "outermost", "mid"]
    if point_selection not in valid_point_selections:
        raise ValueError(f"point_selection must be one of {valid_point_selections}")

    # Get min and max temperatures from data
    min_temp = data.select(pl.min("Temperature")).item()
    max_temp = data.select(pl.max("Temperature")).item()

    # Check exclusion zone is within data range
    if exclusion_lwr <= min_temp:
        raise ValueError(
            f"Lower exclusion bound ({exclusion_lwr}) must \
                be greater than minimum temperature ({min_temp})"
        )
    if exclusion_upr >= max_temp:
        raise ValueError(
            f"Upper exclusion bound ({exclusion_upr}) must \
                be less than maximum temperature ({max_temp})"
        )

    # Count points outside exclusion zone
    n_below = data.filter(pl.col("Temperature") < exclusion_lwr).height
    n_above = data.filter(pl.col("Temperature") > exclusion_upr).height

    # Check if enough points for window size
    if n_below < w:
        raise ValueError(
            f"Not enough points ({n_below}) below exclusion zone for window size {w}"
        )
    if n_above < w:
        raise ValueError(
            f"Not enough points ({n_above}) above exclusion zone for window size {w}"
        )

    # Progress message
    if explicit:
        print(
            f"Detecting endpoints with window size {w}, \
                exclusion zone [{exclusion_lwr}, {exclusion_upr}]"
        )

    # Convert data to numpy arrays for processing
    sorted_data = data.sort("Temperature")
    temps = sorted_data.select("Temperature").to_numpy().flatten()
    dcps = sorted_data.select("dCp").to_numpy().flatten()

    # Fit spline to full data
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_with_gcv(temps, dcps)

    # Calculate residuals
    spline_fit = spline(temps)
    residuals = dcps - spline_fit

    # Create arrays of temperatures, residuals, and indices
    temp_array = temps
    residual_array = residuals
    idx_array = np.arange(len(temps))

    # Split data into regions below and above exclusion zone
    mask_below = temp_array < exclusion_lwr
    mask_above = temp_array > exclusion_upr

    idx_below = idx_array[mask_below]
    idx_above = idx_array[mask_above]

    # Progress message for lower region
    if explicit:
        print("Scanning Lower Region")

    # Calculate variance for each window in lower region
    lower_stats = []
    for i in range(len(idx_below) - w + 1):
        window_indices = idx_below[i : i + w]
        window_residuals = residual_array[window_indices]
        window_std = np.std(window_residuals)
        lower_stats.append((i, window_std))

    # Find window with minimum variance in lower region
    if lower_stats:
        min_var_idx, min_var = min(lower_stats, key=lambda x: x[1])
        # Calculate endpoint index based on point selection
        # For lower region:
        # - Outermost means lowest temperature (closest to beginning of window)
        # - Innermost means highest temperature (closest to end of window)
        if point_selection == "outermost":
            endpoint_offset = 0  # First point in the window (lowest temp)
        elif point_selection == "innermost":
            endpoint_offset = w - 1  # Last point in the window (highest temp)
        else:  # "mid"
            endpoint_offset = w // 2  # Middle point in the window

        lower_endpoint_idx = idx_below[min_var_idx + endpoint_offset]
        lower_endpoint_temp = temp_array[lower_endpoint_idx]
    else:
        # Fallback if no valid windows (shouldn't happen due to earlier checks)
        lower_endpoint_temp = min_temp

    # Progress message for upper region
    if explicit:
        print("Scanning Upper Region")

    # Calculate variance for each window in upper region
    upper_stats = []
    for i in range(len(idx_above) - w + 1):
        window_indices = idx_above[i : i + w]
        window_residuals = residual_array[window_indices]
        window_std = np.std(window_residuals)
        upper_stats.append((i, window_std))

    # Find window with minimum variance in upper region
    if upper_stats:
        min_var_idx, min_var = min(upper_stats, key=lambda x: x[1])
        # Calculate endpoint index based on point selection
        # For upper region:
        # - Outermost means highest temperature (closest to end of window)
        # - Innermost means lowest temperature (closest to beginning of window)
        if point_selection == "outermost":
            endpoint_offset = w - 1  # Last point in the window (highest temp)
        elif point_selection == "innermost":
            endpoint_offset = 0  # First point in the window (lowest temp)
        else:  # "mid"
            endpoint_offset = w // 2  # Middle point in the window

        upper_endpoint_idx = idx_above[min_var_idx + endpoint_offset]
        upper_endpoint_temp = temp_array[upper_endpoint_idx]
    else:
        # Fallback if no valid windows (shouldn't happen due to earlier checks)
        upper_endpoint_temp = max_temp

    # Return results
    return Endpoints(
        lower=float(lower_endpoint_temp),
        upper=float(upper_endpoint_temp),
        method=point_selection,
    )


class EndpointDetector:
    """Class for endpoint detection in thermogram data."""

    def __init__(self) -> None:
        """Initialize EndpointDetector."""
        self.spline_fitter = SplineFitter()

    def detect(
        self,
        data: pl.DataFrame,
        w: int = 90,
        exclusion_lwr: float = 60,
        exclusion_upr: float = 80,
        point_selection: Literal["innermost", "outermost", "mid"] = "innermost",
        explicit: bool = False,
    ) -> Endpoints:
        """
        Detect optimal endpoints for thermogram baseline subtraction.

        Args:
            data: DataFrame with Temperature and dCp columns
            w: Window size for variance calculation
            exclusion_lwr: Lower bound of the exclusion window
            exclusion_upr: Upper bound of the exclusion window
            point_selection: Method for endpoint selection
            explicit: Whether to print progress messages

        Returns:
            Endpoints object with lower and upper endpoints and method used
        """
        return detect_endpoints(
            data, w, exclusion_lwr, exclusion_upr, point_selection, explicit
        )

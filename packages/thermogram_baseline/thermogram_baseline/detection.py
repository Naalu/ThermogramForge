"""Endpoint detection algorithms for thermogram baseline subtraction."""

from typing import Literal, Tuple, Union

import numpy as np
import polars as pl

from .types import Endpoints, ThermogramData


def detect_endpoints(
    data: Union[ThermogramData, pl.DataFrame],
    window_size: int = 90,
    exclusion_lower: float = 60,
    exclusion_upper: float = 80,
    point_selection: Literal["innermost", "outmost", "mid"] = "innermost",
    verbose: bool = False,
) -> Endpoints:
    """
    Detect optimal endpoints for baseline subtraction.

    This function scans through temperature regions outside the exclusion zone,
    looking for windows with minimal variance to establish baseline endpoints.

    Args:
        data: Thermogram data to analyze
        window_size: Number of points to include in each variance window
        exclusion_lower: Lower bound of the exclusion window (temperature)
        exclusion_upper: Upper bound of the exclusion window (temperature)
        point_selection: Method for selecting a point from the minimum variance window
        verbose: Whether to print progress information

    Returns:
        Endpoints object containing the detected lower and upper endpoints
    """
    # Convert input to ThermogramData if it's a DataFrame
    if isinstance(data, pl.DataFrame):
        therm_data = ThermogramData.from_dataframe(data)
    else:
        therm_data = data

    # Extract arrays for easier processing
    temperatures = therm_data.temperature
    values = therm_data.dcp

    # Validate inputs
    if len(temperatures) < window_size * 2:
        raise ValueError(
            f"Not enough data points. Dataset has {len(temperatures)} points, but window size is {window_size}"
        )

    if min(temperatures) > exclusion_lower:
        raise ValueError(
            f"Exclusion zone lower bound ({exclusion_lower}) is below the minimum temperature in data ({min(temperatures)})"
        )

    if max(temperatures) < exclusion_upper:
        raise ValueError(
            f"Exclusion zone upper bound ({exclusion_upper}) is above the maximum temperature in data ({max(temperatures)})"
        )

    # Find indices corresponding to exclusion boundaries
    exclusion_lower_idx = np.searchsorted(temperatures, exclusion_lower)
    exclusion_upper_idx = np.searchsorted(temperatures, exclusion_upper)

    # Check if we have enough points outside the exclusion zone
    if exclusion_lower_idx < window_size:
        raise ValueError(
            f"Not enough points below exclusion zone. Need at least {window_size}, but have {exclusion_lower_idx}"
        )

    if len(temperatures) - exclusion_upper_idx < window_size:
        raise ValueError(
            f"Not enough points above exclusion zone. Need at least {window_size}, but have {len(temperatures) - exclusion_upper_idx}"
        )

    # Scan lower region to find minimum variance window
    if verbose:
        print("Scanning lower temperature region...")

    lower_window_end, _ = _scan_lower_region(
        temperatures, values, window_size, exclusion_lower_idx
    )

    # Scan upper region to find minimum variance window
    if verbose:
        print("Scanning upper temperature region...")

    upper_window_start, _ = _scan_upper_region(
        temperatures, values, window_size, exclusion_upper_idx
    )

    # Select specific points based on the selection method
    lower_window_start = lower_window_end - window_size + 1
    upper_window_end = upper_window_start + window_size - 1

    lower_point_idx = _select_point_from_window(
        lower_window_start, lower_window_end, point_selection, low_region=True
    )

    upper_point_idx = _select_point_from_window(
        upper_window_start, upper_window_end, point_selection, low_region=False
    )

    # Get temperatures at the selected indices
    lower_endpoint = temperatures[lower_point_idx]
    upper_endpoint = temperatures[upper_point_idx]

    if verbose:
        print(f"Selected endpoints: lower={lower_endpoint}, upper={upper_endpoint}")

    return Endpoints(lower=lower_endpoint, upper=upper_endpoint, method=point_selection)


def _scan_lower_region(
    temperatures: np.ndarray,
    values: np.ndarray,
    window_size: int,
    exclusion_idx: int,
) -> Tuple[int, float]:
    """
    Scan the lower temperature region to find the window with minimum variance.

    Args:
        temperatures: Array of temperature values
        values: Array of dCp values
        window_size: Size of the window to analyze
        exclusion_idx: Index where exclusion zone begins

    Returns:
        Tuple of (window_end_index, minimum_variance)
    """
    min_variance = float("inf")
    best_window_end = window_size - 1

    # Scan through all possible windows
    for i in range(window_size - 1, exclusion_idx):
        window_start = i - window_size + 1
        window_end = i

        # Calculate variance of window
        window_variance = np.var(values[window_start : window_end + 1])

        # Update if this window has lower variance
        if window_variance < min_variance:
            min_variance = window_variance
            best_window_end = window_end

    return best_window_end, min_variance


def _scan_upper_region(
    temperatures: np.ndarray,
    values: np.ndarray,
    window_size: int,
    exclusion_idx: int,
) -> Tuple[int, float]:
    """
    Scan the upper temperature region to find the window with minimum variance.

    Args:
        temperatures: Array of temperature values
        values: Array of dCp values
        window_size: Size of the window to analyze
        exclusion_idx: Index where exclusion zone ends

    Returns:
        Tuple of (window_start_index, minimum_variance)
    """
    min_variance = float("inf")
    best_window_start = exclusion_idx
    data_length = len(temperatures)

    # Scan through all possible windows
    for i in range(exclusion_idx, data_length - window_size + 1):
        window_start = i
        window_end = i + window_size - 1

        # Calculate variance of window
        window_variance = np.var(values[window_start : window_end + 1])

        # Update if this window has lower variance
        if window_variance < min_variance:
            min_variance = window_variance
            best_window_start = window_start

    return best_window_start, min_variance


def _select_point_from_window(
    start_idx: int,
    end_idx: int,
    method: Literal["innermost", "outmost", "mid"],
    low_region: bool = True,
) -> int:
    """
    Select a specific point from a window based on the selection method.

    Args:
        start_idx: Start index of the window
        end_idx: End index of the window
        method: Method for selecting the point
        low_region: Whether this is the lower temperature region

    Returns:
        Index of the selected point
    """
    if method == "innermost":
        # Select point closest to center of data
        return end_idx if low_region else start_idx

    elif method == "outmost":
        # Select most extreme point
        return start_idx if low_region else end_idx

    elif method == "mid":
        # Select middle point
        return start_idx + (end_idx - start_idx) // 2

    else:
        raise ValueError(f"Unknown point selection method: {method}")

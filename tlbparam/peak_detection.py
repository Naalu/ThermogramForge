"""
Peak detection module for thermogram data analysis.

This module implements peak detection and characterization for thermogram data,
locating peaks in specific temperature ranges and calculating their properties.
"""

from typing import Dict, Tuple

import numpy as np
import polars as pl


def gen_peak(
    x: np.ndarray, temperatures: np.ndarray, peak_range: Tuple[float, float]
) -> Dict[str, float]:
    """
    Find peak in a specific temperature range.

    This function replicates the R gen_peak function, finding the
    maximum value within a specified temperature range.

    Args:
        x: Array of dCp values
        temperatures: Array of temperature values
        peak_range: Tuple of (min_temp, max_temp)

    Returns:
        Dictionary with peak height and temperature
        {
            'peak_height': float,
            'peak_temp': float
        }
    """
    # Validate inputs
    if len(x) != len(temperatures):
        raise ValueError("x and temperatures must have the same length")

    # Find indices where temperatures are within range
    min_temp, max_temp = peak_range
    in_range = (temperatures >= min_temp) & (temperatures <= max_temp)

    # If no points in range, return zeros
    if not np.any(in_range):
        return {"peak_height": 0.0, "peak_temp": 0.0}

    # Find maximum value within range
    temp_in_range = temperatures[in_range]
    x_in_range = x[in_range]

    max_idx = np.argmax(x_in_range)
    peak_height = float(x_in_range[max_idx])
    peak_temp = float(temp_in_range[max_idx])

    return {"peak_height": peak_height, "peak_temp": peak_temp}


def gen_fwhm(x: np.ndarray, temperatures: np.ndarray) -> float:
    """
    Calculate full width at half maximum.

    This function replicates the R gen_fwhm function, calculating the
    width of a peak at half its maximum height.

    Args:
        x: Array of dCp values
        temperatures: Array of temperature values

    Returns:
        FWHM value
    """
    # Validate inputs
    if len(x) != len(temperatures):
        raise ValueError("x and temperatures must have the same length")

    if len(x) < 3:
        raise ValueError("At least 3 data points are required for FWHM calculation")

    # Find maximum value and its position
    ymax = np.max(x)
    ymin = np.min(x)
    lymax = np.argmax(x)

    # Calculate half-maximum height
    half = (ymax - ymin) / 2 + ymin

    # Find points closest to half-maximum on each side of peak
    # Left side (from start to max)
    left_values = x[: lymax + 1]
    left_temps = temperatures[: lymax + 1]

    if len(left_values) > 0:
        left_diff = np.abs(left_values - half)
        x1 = np.argmin(left_diff)
        temp1 = left_temps[x1]
    else:
        # If no left side, use the first point
        temp1 = temperatures[0]

    # Right side (from max to end)
    right_values = x[lymax:]

    if len(right_values) > 0:
        right_diff = np.abs(right_values - half)
        # Need to add lymax to get the correct index in original array
        x2 = lymax + np.argmin(right_diff)
        temp2 = temperatures[x2]
    else:
        # If no right side, use the last point
        temp2 = temperatures[-1]

    # Calculate width
    fwhm = abs(temp2 - temp1)

    return float(fwhm)


class PeakDetector:
    """Class for detecting and characterizing peaks in thermogram data."""

    def __init__(self) -> None:
        """Initialize PeakDetector."""
        pass

    def detect_peaks(
        self, data: pl.DataFrame, temp_col: str = "Temperature", value_col: str = "dCp"
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect peaks in thermogram data.

        Args:
            data: DataFrame with thermogram data
            temp_col: Name of temperature column
            value_col: Name of value column

        Returns:
            Dictionary with peak information
        """
        # Validate inputs
        if not all(col in data.columns for col in [temp_col, value_col]):
            raise ValueError(
                f"Data must contain '{temp_col}' and '{value_col}' columns"
            )

        # Extract data
        temperatures = data.select(pl.col(temp_col)).to_numpy().flatten()
        values = data.select(pl.col(value_col)).to_numpy().flatten()

        # Define peak ranges based on standard temperature regions
        peak_ranges = {
            "Peak 1": (60.0, 66.0),
            "Peak 2": (67.0, 73.0),
            "Peak 3": (73.0, 81.0),
            "Peak F": (50.0, 54.0),
        }

        # Detect each peak
        peaks = {}
        for peak_name, peak_range in peak_ranges.items():
            peak_info = gen_peak(values, temperatures, peak_range)
            peaks[peak_name] = peak_info

        # Calculate FWHM for the maximum peak
        try:
            fwhm = gen_fwhm(values, temperatures)
            peaks["FWHM"] = {"fwhm": fwhm}
        except ValueError:
            # Handle the case where FWHM cannot be calculated
            peaks["FWHM"] = {"fwhm": 0.0}

        return peaks
